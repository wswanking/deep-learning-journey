# 第26章：可审计检索与生成对照

数据、文档和问答均为原创虚构设施记录，所有数值不对应现实服务。12篇文档可独立用于检索；验证6题，测试12个有答案问题及4个不存在实体问题。数据生成源data.py与三份JSONL同时提供。

## 运行

先按第21章说明生成基座，再从仓库根目录运行：

```powershell
& ./.venv/Scripts/python.exe examples/26-retrieval-augmented-generation/run.py
& ./.venv/Scripts/python.exe examples/26-retrieval-augmented-generation/figures.py
```

--skip-lm只做检索，不能代替完整配对生成。输出在build/experiments/26-retrieval-augmented-generation/，不会向Git提交权重或缓存。

## 协议与实测

- Windows Python3.11.5/NumPy2.4.6/Torch2.15.0+cpu，检索float64、生成float32/1线程。
- 字符三元组的平滑TF-IDF与L2归一化，精确扫描。选择6组切分的验证EM，冻结后测试。选72字符/重叠0/k3/最多2块，24×280矩阵、53760数值字节。
- 第一次完整实测：234断言通过；含导入2.732秒，检索与核验0.048秒，配对生成0.173秒；峰值内存未测。
- 受限实体/字段规则读者12/12正确、4/4拒答、12/12引用通过；测试不同切分/top1/无重排全部相同，不能宣称优越性。
- 真实21基座有无完整检索证据均0/12 EM。保留全部生成字节和非法UTF8标记；规则读者不是语言模型。
- 基座SHA256为326e1a5e405fc3280aae20e57f3eadd7ef0d7f54df3cb35af6a87bc679068e5a，报告实际计算指纹，复跑须核对。
- 没有训练DPR/交叉编码器，没有下载外部数据，也没有近似索引加速声称。

## 文件与可复现边界

data.py生成documents/validation/test三份JSONL；retrieval.py保存原文区间、精确向量、受限读者和评价；run.py建立索引、实际恢复NPZ并生成报告；figures.py重建五幅真SVG。正文给出符号、手算、来源与每个对照的局限。

原文快照、特征顺序、IDF和切分一起确定索引；index.npz实际逐项恢复一致。results.json包含全部候选/上下文/回答/引用和模型配对输出，便于27章分类失败。时间依赖机器负载，不保证跨环境逐位一致。
