# V6.3

1. `EngineConfig.max_active_seqs` 默认值由 128 改为 372。
2. `example.py --max-active-seqs` 默认值和帮助文本同步为 372；`bench.py` 继承该参数。
3. 包版本设为 0.6.3，运行消息标识更新为 V6.3，基准默认文件名改为 benchmark-v63.json。
4. 单次调度上限 128、Target Decode Graph 桶、28/8 GiB KV 配额和全部调度算法沿用 V6.2。
5. 保留 V6.2 原目录，单独创建 V6.3 源码与 ZIP；不运行模型或操作 Colab。
