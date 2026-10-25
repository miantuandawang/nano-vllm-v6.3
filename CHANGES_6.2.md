# V6.2

1. Target decode Graph 固定捕获原生桶1/2/4/8/16/32/48/64/80/96/112/128；复用已有捕获、padding和replay函数。
2. `batch` 默认128、允许1–128；`max_active_seqs` 默认128，独立限制活跃请求容量。
3. Draft decode与catch-up明确限制最大16，Target Verify沿用小批次上限，避免Target扩桶同时扩大Draft图。
4. 沿用V6.1：只有调度投机时才执行请求级Draft Prefill或补算，Target-only阈值默认16。
5. CLI、版本号、文档和已有Graph测试期望同步更新。按用户要求不运行任何测试。

V6.1原目录未覆盖；V6.1历史CPU/CUDA验证报告未作为V6.2报告复制。
KV配额与Graph内存预留估计沿用V6.1，128桶的运行行为、峰值与吞吐本次未验证。
