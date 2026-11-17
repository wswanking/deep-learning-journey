# 第17章 CPU实验

从正式仓库根目录运行：

```powershell
& ./.venv/Scripts/python.exe -B examples/17-representation-learning/run.py
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python -B examples/17-representation-learning/run.py
```

- 无网络下载；训练/验证/测试128/64/64个三维合成点，种子20261117，均值只从训练集拟合。
- PCA取1、2、3个方向；AE为3→16Tanh→r→16Tanh→3，每个r使用11、22、33种子。
- 每次1200步全批次Adam，float64 CPU单线程；验证每20步选权重，全部方案冻结后统一测试。
- 额外复跑r1种子11，核对整条曲线和恢复预测。检查点只支持推理加载，未声称精确续训。
- `--steps`允许学习者另做实验，改变后指标不再对应正文；`--skip-figures`保留已提交SVG。
- 输出：`build/experiments/17-representation-learning/results.json`与`checkpoints/`；默认读取真实报告生成纯SVG。
- 单独重画：运行同目录`figures.py`。
- 当前实测43核对通过，训练/检查8.728秒；原始报告包含全部数字和环境，时间会随机器负载变化。
- 模型、缓存和生成文档在忽略的build/内，不提交权重或临时图。正文见`docs/chapters/17-representation-learning.md`。
