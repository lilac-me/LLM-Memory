# 训练显存建模

基于 2.8T 工作表的交互式训练显存模型。

支持 MLA、KDA、标准/latent MoE，FSDP × EP 分片，BF16/FP32 梯度，Adam 与细粒度激活 swap，预取峰值、DDR、IO 和 MFU 估算。

## GitHub Pages

发布入口为根目录 index.html。

Settings → Pages → Deploy from a branch → main / (root) → Save。

启用后访问：https://lilac-me.github.io/LLM-Memory/

本页无需后端服务，计算在浏览器中进行。页面中的建模假设、原表修正与计算覆盖范围可在“口径与来源”中查看。
