#!/usr/bin/env python3
"""Ascend communication audit: streaming trace + kernel CSV -> local evidence JSON/CSV.

python profiling_parser.py --trace trace_view.json --kernels kernel_details.csv --output-dir comm_audit
Large traces require ijson (`python -m pip install ijson`). Inputs are read-only.
Bandwidth is payload / collective duration, including waits, NOT a measured physical link rate.
"""
import argparse,bisect,csv,json,math,re,statistics,time
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path

VERSION='atlas-communication-profile-v1'
DTYPES={'BF16':2,'BFP16':2,'BFLOAT16':2,'FLOAT16':2,'FP16':2,'HALF':2,'FLOAT':4,'FLOAT32':4,'FP32':4,'DOUBLE':8,'FP64':8,'INT8':1,'UINT8':1,'INT16':2,'UINT16':2,'INT32':4,'UINT32':4,'INT64':8,'UINT64':8,'BOOL':1}
def dtype(s):
 s=str(s).strip().upper().removeprefix('DT_').removeprefix('HCCL_DATA_TYPE_')
 return ('BF16' if s in ('BFP16','BFLOAT16') else 'FP32' if s in ('FLOAT','FLOAT32') else s),DTYPES.get(s)
def shapes(s):
 out=[]
 for item in str(s).strip().strip('"').split(';'):
  item=item.strip().strip('[]()').replace(' ','')
  out.append([int(x) for x in item.split(',')] if re.fullmatch(r'\d+(,\d+)*',item) else None)
 return out
def op_name(name):
 s=re.sub('[^a-z]','',str(name).lower())
 for k,v in [('alltoall','AllToAll'),('reducescatter','ReduceScatter'),('allgather','AllGather'),('allreduce','AllReduce'),('receive','Recv'),('recv','Recv'),('send','Send'),('broadcast','Broadcast')]:
  if k in s:return v
 return None
def iterate(path):
 if path.suffix=='.jsonl':
  with path.open() as f:
   for line in f:
    if line.strip():yield json.loads(line,parse_float=Decimal)
  return
 with path.open('rb') as f:first=f.read(4096).lstrip(b'\xef\xbb\xbf \r\n\t')[:1]
 try:import ijson
 except ImportError:
  if path.stat().st_size>64*2**20:raise ValueError('Trace > 64 MiB: install ijson for streaming; refusing a full json.load')
  with path.open(encoding='utf-8-sig') as f:obj=json.load(f,parse_float=Decimal)
  yield from obj if isinstance(obj,list) else obj['traceEvents']
 else:
  with path.open('rb') as f:
   if f.read(3)!=b'\xef\xbb\xbf':f.seek(0)
   yield from ijson.items(f,'item' if first==b'[' else 'traceEvents.item')
def merge(intervals):
 out=[]
 for a,b in sorted(intervals):
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return out
def span(intervals):return sum(b-a for a,b in merge(intervals))
def intersect(a,b):
 a,b=merge(a),merge(b);i=j=0;out=[]
 while i<len(a) and j<len(b):
  lo=max(a[i][0],b[j][0]);hi=min(a[i][1],b[j][1])
  if hi>lo:out.append([lo,hi])
  if a[i][1]<b[j][1]:i+=1
  else:j+=1
 return span(out)
def quantile(xs,p):
 xs=sorted(xs);x=(len(xs)-1)*p;i=int(x)
 return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(x-i)
def read_kernels(path,device=None,start=None,end=None,model_config=None):
 candidates=[];compute=[];origins=[];name_counts=Counter();all_collectives=[];optimizer_starts=[]
 model_config=model_config or {};moe_width=model_config.get("moe_intermediate_size");hidden=model_config.get("hidden_size")
 with path.open(newline='',encoding='utf-8-sig') as f:
  reader=csv.DictReader(f)
  required={'Start Time(us)','Duration(us)','Type','Output Shapes','Output Data Types'}
  if not required.issubset(reader.fieldnames or []):raise ValueError('Unsupported kernel CSV headers; expected Ascend kernel_details.csv with explicit us units')
  for lineno,r in enumerate(reader,2):
   dev=r.get('Device_id',r.get('Device ID','')).strip()
   if device is not None and dev!=str(device):continue
   ts=Decimal(r['Start Time(us)'].strip());dur=Decimal(r['Duration(us)'].strip())
   if start is not None and ts<start or end is not None and ts+dur>end:continue
   origins.extend([ts,ts+dur]) if not origins else None
   if origins:origins=[min(origins[0],ts),max(origins[-1],ts+dur)]
   ty=r['Type'];name=r.get('Name',ty);name_counts[ty]+=1
   if 'ApplyAdam' in ty or 'ApplyAdam' in name:optimizer_starts.append(ts)
   if not ('Hccl' in ty or ty.startswith('hcom')) and any(x in r.get('Accelerator Core','') for x in ['AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV']):compute.append((ts,ts+dur))
   if ty.startswith('hcom_'):
    all_collectives.append({'device':dev,'stream':r.get('Stream ID'),'task':r.get('Task ID'),'ts':ts,'dur':dur,'name':name})
   ins=shapes(r.get('Input Shapes',''));outs=shapes(r['Output Shapes']);dts=r['Output Data Types'].split(';')
   kind=None
   if 'permute' in name.lower() and 'unpermute' not in name.lower():kind='permute_named'
   elif ty in ('GatherV2','GatherElements','Index','ScatterElementsV2','ScatterAdd'):kind='routing_candidate'
   elif ty=='GroupedMatmul' and len(ins)>1 and ins[0] and ins[1] and outs[0] and len(ins[0])==2 and len(ins[1])==3 and len(outs[0])==2:
    # FC2-like shape: inner dim -> larger hidden output. This is a candidate, not a phase proof.
    if outs[0][0]==ins[0][0] and outs[0][-1]>ins[0][-1]:
     kind='fc2_shape_candidate' if moe_width and hidden and ins[0][-1]==moe_width and outs[0][-1]==hidden else 'grouped_matmul_output_candidate'
   if not kind:continue
   for oi,shape in enumerate(outs):
    if not shape or len(shape)!=2:continue # exclude index vectors and expert weight-gradient tensors
    dt,db=dtype(dts[oi] if oi<len(dts) else '')
    if dt not in ('BF16','FP16','FLOAT16','FP32'):continue
    candidates.append({'id':f'csv:{lineno}:out{oi}','kind':kind,'name':name,'type':ty,'device':dev,'stream':r.get('Stream ID'),'task':r.get('Task ID'),'ts':ts,'dur':dur,'end':ts+dur,'shape':shape,'dtype':dt,'dtype_bytes':db,'bytes':math.prod(shape)*db,'csv_line':lineno,'output_index':oi})
 if not origins:raise ValueError('No kernel records in the selected device/window')
 return candidates,compute,origins,name_counts,all_collectives,optimizer_starts

def scan_trace(path,candidates,start,end,trace_unit='us',device=None):
 mul={'us':Decimal(1),'ns':Decimal('.001'),'ms':Decimal(1000),'s':Decimal(1000000)}[trace_unit]
 procs={};labels={};lanes={};comms=[];joins=defaultdict(list);wanted=defaultdict(list);step_markers=[]
 for p in candidates:wanted[(str(p['stream']),str(p['task']))].append(p)
 tick=time.monotonic()
 for i,e in enumerate(iterate(path),1):
  if i%1000000==0:print(f'trace: {i:,} events, {time.monotonic()-tick:.1f}s',flush=True)
  args=e.get('args') or {};ph=e.get('ph');name=e.get('name','');pid=e.get('pid');tid=e.get('tid')
  if ph=='M':
   if name=='process_name':procs[pid]=args.get('name','')
   elif name=='process_labels':labels[pid]=args.get('labels','')
   elif name=='thread_name':lanes[(pid,tid)]=args.get('name','')
   continue
  if ph!='X' or 'ts' not in e:continue
  ts=Decimal(str(e['ts']))*mul;dur=Decimal(str(e.get('dur',0)))*mul
  if ts<start or ts+dur>end:continue # complete events only: never divide full bytes by a clipped time
  if re.search(r'ProfilerStep|train_step',name,re.I):step_markers.append({'name':name,'ts':str(ts),'duration_us':float(dur)})
  cid=args.get('connection_id',args.get('connectionId'))
  task=args.get('Task Id',args.get('taskId'));key=(str(tid),str(task))
  if key in wanted and task is not None:
   for p in wanted[key]:
    if abs(ts-p['ts'])<=Decimal('.1'):joins[p['id']].append({'pid':pid,'connection_id':cid})
  op=op_name(name)
  # Group wrappers have rank_size, unlike Host APIs, Hardware kernels, or Plane sub-tasks.
  if not op or 'rank_size' not in args:continue
  comms.append({'op':op,'pid':pid,'tid':tid,'name':name,'ts':ts,'dur':dur,'group_size':int(args['rank_size']),'raw_count':args.get('count'),'dtype':args.get('data_type',''),'connection_id':cid,'algorithm':args.get('alg_type','')})
 def wanted_device(pid):
  if device is None:return True
  label=labels.get(pid,'');return bool(re.search(r'\b(?:NPU|Device)\s*'+re.escape(str(device))+r'\b',label,re.I))
 for p in candidates:
  matched=[x for x in joins[p['id']] if wanted_device(x['pid']) and 'Hardware' in procs.get(x['pid'],'')]
  if len(matched)==1:p['connection_id']=matched[0]['connection_id']
 for x in comms:x['group']=lanes.get((x['pid'],x['tid']),str(x['tid']));x['device_label']=labels.get(x['pid'],'unknown')
 selected=[x for x in comms if wanted_device(x['pid']) and procs.get(x['pid'])=='Communication' and x['group'].startswith('Group ')]
 if not selected:raise ValueError('No Communication group wrappers found; confirm trace, device metadata, window and profiler level')
 return selected,step_markers,i

def analyze(args):
 t0=time.monotonic();start=Decimal(args.start_us) if args.start_us else None;end=Decimal(args.end_us) if args.end_us else None
 model=json.loads(args.model_config.read_text()) if getattr(args,'model_config',None) else {}
 candidates,compute,bounds,types,hardware,optimizer_starts=read_kernels(args.kernels,args.device,start,end,model)
 lo=start if start is not None else bounds[0];hi=end if end is not None else bounds[1]
 if hi<=lo:raise ValueError('Window start must precede end')
 print(f'CSV: {len(candidates)} shape candidates; window {lo} .. {hi}',flush=True)
 comms,steps,event_count=scan_trace(args.trace,candidates,lo,hi,args.trace_unit,args.device)
 # Candidate search bounded by dtype/numel and preceding device completion. No asserted data dependency.
 by_size=defaultdict(list)
 for p in candidates:by_size[(p['bytes'],p['dtype'])].append(p)
 for key in by_size:by_size[key].sort(key=lambda p:p['end'])
 by_times={key:[p['end'] for p in ps] for key,ps in by_size.items()}
 output=[]
 for e in sorted(comms,key=lambda x:x['ts']):
  dt,db=dtype(e['dtype']);count=int(e['raw_count']) if e['raw_count'] is not None else None;g=e['group_size'];op=e['op'];logical=None;sem='unresolved';evidence='unresolved';peers=[]
  if count is not None and db:
   if op in ('AllGather','ReduceScatter') and args.fsdp_count=='api-shard':logical=count*db*g;sem='sendCount/recvCount shard × group (HCCL API assumption)';evidence='api_count_assumption'
   elif op in ('AllGather','ReduceScatter') and args.fsdp_count=='full':logical=count*db;sem='count is full tensor (explicit override)';evidence='count_semantics_override'
   elif op in ('AllReduce','Send','Recv','Broadcast'):logical=count*db;sem='tensor count';evidence='count_reported'
   elif op=='AllToAll':
    raw_bytes=count*db;key=(raw_bytes,dt);ps=by_size.get(key,[]);times=by_times.get(key,[]);idx=bisect.bisect_right(times,e['ts']);left=bisect.bisect_left(times,e['ts']-Decimal(str(args.max_gap_us)))
    peers=ps[max(left,idx-3):idx]
    if args.alltoall_count=='send-total':logical=raw_bytes;sem='sum(sendCounts) (explicit override)';evidence='count_semantics_override'
    elif peers:logical=raw_bytes;sem='count × dtype agrees with nearby output; total-send interpretation provisional';evidence='shape_count_agreement_candidate'
  phase_hint='post_optimizer_gather_candidate' if op=='AllGather' and optimizer_starts and e['ts']>=min(optimizer_starts) else 'unresolved'
  normalized={'phase_hint':phase_hint,'id':f'comm:{len(output)}','op':op,'name':e['name'],'device':e['device_label'],'group':e['group'],'group_size':g,'dtype':dt,'dtype_bytes':db,'raw_count':count,'count_semantics':sem,'evidence':evidence,'algorithm':e['algorithm'],'connection_id':e['connection_id'],'start_us':str(e['ts']),'relative_start_us':float(e['ts']-lo),'duration_us':float(e['dur']),'logical_bytes':logical,'shape_candidates':[]}
  for p in reversed(peers):normalized['shape_candidates'].append({k:p.get(k) for k in ['id','kind','name','shape','dtype','bytes','csv_line','stream','task','connection_id','output_index']}|{'gap_us':float(e['ts']-p['end']),'relation':'shape/dtype/time candidate; dependency not proven'})
  normalized['logical_gbps']=logical/float(e['dur'])/1000 if logical is not None and e['dur']>0 else None
  # Ring equivalent is diagnostic, never claimed as observed HCCL fabric traffic.
  factor=(2 if op=='AllReduce' else 1)*(g-1)/g if g>1 else 0
  if op in ('Send','Recv'):factor=1
  normalized['equivalent_send_bytes']=logical*factor if logical is not None and op!='Broadcast' else None
  normalized['equivalent_gbps']=normalized['equivalent_send_bytes']/float(e['dur'])/1000 if normalized['equivalent_send_bytes'] is not None and e['dur']>0 else None
  output.append(normalized)
 grouped=defaultdict(list)
 for e in output:grouped[(e['op'],e['device'],e['group'],e['group_size'],e['dtype'],e['logical_bytes'],e['raw_count'],e['evidence'],e['phase_hint'])].append(e)
 groups=[]
 for key,events in grouped.items():
  op,device,group,g,dt,n,raw_count,evidence,phase_hint=key;durations=[x['duration_us'] for x in events];duration=sum(durations);ints=[(x['relative_start_us'],x['relative_start_us']+x['duration_us']) for x in events]
  total=n*len(events) if n is not None else None
  groups.append({'phase_hint':phase_hint,'id':f'group:{len(groups)}','op':op,'device':device,'group':group,'group_size':g,'dtype':dt,'dtype_bytes':events[0]['dtype_bytes'],'logical_bytes_per_call':n,'raw_count':raw_count,'calls':len(events),'duration_sum_us':duration,'duration_union_us':span(ints),'p50_us':statistics.median(durations),'p95_us':quantile(durations,.95),'max_us':max(durations),'logical_gbps':total/duration/1000 if total is not None and duration else None,'equivalent_gbps':sum(e['equivalent_send_bytes'] for e in events)/duration/1000 if total is not None and events[0]['equivalent_send_bytes'] is not None and duration else None,'evidence':evidence,'count_semantics':events[0]['count_semantics'],'shape_candidate_calls':sum(bool(e['shape_candidates']) for e in events),'calibration_requires_confirmation':True})
 groups.sort(key=lambda x:-x['duration_sum_us'])
 commints=[(float(x['ts']-lo),float(x['ts']+x['dur']-lo)) for x in comms];compints=[(float(s-lo),float(e-lo)) for s,e in compute]
 union=span(commints);overlap=intersect(commints,compints)
 shape_summary=Counter((p['kind'],tuple(p['shape']),p['dtype'],p['bytes']) for p in candidates)
 warnings=['AllToAll 输出与通信的 shape/时间一致仅构成候选关联；没有指针/框架 flow 时不能宣称依赖已证明。','AllGather/ReduceScatter 的原始 count 不等于完整层大小；按所选 count 语义换算，不再除 EP。','有效带宽含集合算子内等待，不等于物理链路吞吐；聚合速率用 Σbytes/Σduration，不平均各调用的 GB/s。','采样窗口不自动当作完整 step；记录设备、group、dtype、消息大小后才可回填。','当前导出未将 Send/Recv 混入 FSDP 或 AllToAll；collective 子 task 和 Hardware 重复副本不参与通信量求和。']
 if any(g['phase_hint']=='post_optimizer_gather_candidate' for g in groups):warnings.append('发现优化器更新开始后的 AllGather；可能是 optimizer bucket 参数同步，不能直接标记为一层 FSDP gather。')
 if not any('permute' in t.lower() for t in types):warnings.append('CSV 没有具名 permute；Gather/Scatter 只能作路由实现候选。')
 report={'schema':VERSION,'source':{'trace':args.trace.name,'kernels':args.kernels.name,'device':args.device,'trace_events_scanned':event_count,'model_config':getattr(args,'model_config',None).name if getattr(args,'model_config',None) else None,'parse_seconds':round(time.monotonic()-t0,3),'fsdp_count_semantics':args.fsdp_count,'alltoall_count_semantics':args.alltoall_count},'window':{'start_us':str(lo),'end_us':str(hi),'span_us':float(hi-lo),'step_boundary_verified':False,'step_markers':steps,'selection':'complete events only'},'metrics':{'compute_source':'kernel CSV AI/MIX core events (not allocator or host time)','optimizer_first_relative_us':float(min(optimizer_starts)-lo) if optimizer_starts else None,'collective_calls':len(output),'collective_duration_sum_us':sum(e['duration_us'] for e in output),'collective_union_us':union,'compute_union_us':span(compints),'compute_overlap_us':overlap,'comm_without_compute_us':union-overlap},'groups':groups,'shape_summary':[{'kind':k[0],'shape':list(k[1]),'dtype':k[2],'bytes':k[3],'calls':v} for k,v in shape_summary.most_common()],'events':output,'warnings':warnings}
 args.output_dir.mkdir(parents=True,exist_ok=True)
 (args.output_dir/'communication_profile.json').write_text(json.dumps(report,ensure_ascii=False,separators=(',',':')))
 for name,rows in [('communication_groups',groups),('communication_events',output)]:
  with (args.output_dir/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
 lines=['# 通信 Profiling 核验','',f'窗口 {float(hi-lo)/1e6:.6f} s；{len(output)} 次通信；解析 {report["source"]["parse_seconds"]} s。','', '| 算子 | group | dtype | count | 单次逻辑 MiB | 次数 | 时长和 s | P50 ms | P95 ms | 有效 GB/s | 证据 |','|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
 for g in groups:
  n=g['logical_bytes_per_call'];bw=g['logical_gbps'];lines.append(f"| {g['op']} | {g['group_size']} | {g['dtype']} | {g['raw_count']} | {n/2**20 if n is not None else '待确认'} | {g['calls']} | {g['duration_sum_us']/1e6:.6f} | {g['p50_us']/1000:.4f} | {g['p95_us']/1000:.4f} | {round(bw,3) if bw is not None else '待确认'} | {g['evidence']} |")
 lines.extend(['','## 限制','']+['- '+w for w in warnings]);(args.output_dir/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'output':str(args.output_dir),'groups':len(groups),'events':len(output),'seconds':report['source']['parse_seconds'],'metrics':report['metrics']},ensure_ascii=False),flush=True)
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True);p.add_argument('--kernels',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='0');p.add_argument('--model-config',type=Path,help='Optional model config to distinguish FC2-like grouped GEMMs');p.add_argument('--start-us');p.add_argument('--end-us');p.add_argument('--trace-unit',choices=['us','ns','ms','s'],default='us');p.add_argument('--max-gap-us',type=float,default=5000);p.add_argument('--fsdp-count',choices=['api-shard','full','unknown'],default='api-shard');p.add_argument('--alltoall-count',choices=['shape-check','send-total'],default='shape-check');args=p.parse_args();analyze(args)
if __name__=='__main__':main()
