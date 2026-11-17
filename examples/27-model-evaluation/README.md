# 第27章实践：冻结模型评估与失败分析

在完成21、22、26真实产物后，于仓库根目录执行：

```powershell
& ./.venv/Scripts/python.exe examples/27-model-evaluation/run.py --write-figures
```

CPU单线程、模型float32、统计float64，不训练、不下载模型或外部语料。缺前章检查点或RAG报告时明确报错。完整原理见[正文](../../docs/chapters/27-model-evaluation.md)。

## 固定数据与协议

`evaluation.jsonl`包含第22章24条既有测试复评，以及本章原创g..l字母复制/反转12条诊断。既有测试不是新盲测；新字母也不是开放基准。`data.py`核对源文件与声明规则、ID唯一及不与SFT训练/验证指令重合。没有按这些输出修改模型。

基座与SFT共用21接口、同一模板，贪心最多8新字节，LF停止。严格EM要求输出参考两个字母+LF；格式为两个a..l字母+LF；另报UTF8/停止/NLL。互斥分类优先级correct→invalid_utf8→no_stop→format_invalid→wrong_answer，原始布尔标记也保留。原样输出基线不理解反转，平衡任务中恰好50%。

27项核对包括RAG冻结协议重新计算、逐个引用原文位置、手算指标/校准、来源未改及分类计数。按语义组同时重抽模型差值，2000次seed2701，只做小样本描述范围。虚构校准手算不是LM校准实测。没有人类标注；可参考[人工评价表](human-rubric.md)准备后续评价。

## 产物接口

`build/experiments/27-model-evaluation/results.json`使用schema_version1：

- `data`、`protocol`、`sources`：数据SHA、模板/停止/计分、冻结输入来源与元数据。
- `language_models.summaries`：每模型每切片分母、EM/格式/NLL、失败计数与计时。
- `language_models.outputs`：以case ID索引全部字节、文本、状态与分类。
- `language_models.paired_group_comparison`：配对组差与重抽范围。
- `rag`：来源快照指纹、规则与真实LM的独立轨迹、显式删证/零上下文干预。
- `hand`：原创混淆矩阵、Brier与ECE手算。
- `checks`、`resources`：实际核对与范围；无进程峰值声明。

同时保存输入的base/sft冻结副本和26原始报告快照。二进制和运行报告在忽略目录重建；代码、数据与七SVG为追溯源。`figures.py`单独执行只从JSON画图，不运行模型。图形刻度随实际延迟自动覆盖所有点。

## 真实结果与限制

2026-11-17复跑：基座两套EM均0；SFT既有6/24、域外0/12、格式均100%；原样输出12/24与6/12。与2026-11-15准确率/NLL/分类一致，耗时不保证一致。RAG规则读者12有支持回答+4正确拒答，真实35k模型12题有证据仍错，不能用规则结果冒充LM能力。27项检查通过；无训练步数、学习率或新增参数。

单次小样本CPU时间仅供本协议诊断，不外推服务P95或并发性能；参数负载141312B也不是峰值内存。组bootstrap仅3/6个合成组，不提供通用能力的总体覆盖保证。
