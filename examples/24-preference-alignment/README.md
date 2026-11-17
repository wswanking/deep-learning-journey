# 第24章实践：固定SFT参考的CPU DPO

先运行第21章与第22章，确保真实`build/experiments/22-instruction-tuning/response_only.pt`存在。在仓库根目录运行：

```powershell
& ./.venv/Scripts/python.exe examples/24-preference-alignment/run.py --write-figures
```

Linux / WSL（先按[Linux环境指南](../../docs/linux-guide.md)准备依赖）：

```bash
.venv-linux/bin/python examples/24-preference-alignment/run.py --write-figures
```

默认CPU、float32、1线程，无外部模型/数据下载。复用21模型API及22冻结SFT；缺文件报错，不替换随机模型。完整原理见[章节正文](../../docs/chapters/24-preference-alignment.md)。

## 数据与协议

`data.py`定义并校验原创g..l字母首/末选择规则。三份`data/*.jsonl`共60条比较，15个无序字母组，seed2400按9/3/3组划分，36/12/12条；同一组正反顺序不跨集。chosen/rejected各为一个字母，计分时均加LF，恰好两个回答目标。与22的a..f复制/反转指令隔离。没有人类标注，不称人类胜率。

每组35,328参数；固定同一SFT初始与参考，训练全部参数。三个预声明配置β1/β0.1/β1翻转全部训练标签。AdamW lr0.0005、wd0、clip1，batch12对，300步，每组3600比较对（重复），采样seed2401、运行seed2402。最终固定300步，验证仅诊断；测试所有冻结模型各一次，未以测试反馈调参。

回答概率为log_softmax/gather/response mask/求和，提示不计分但保留上下文，补齐键独立mask。DPO稳定softplus，参考无梯度缓存。检查teacher-forcing与逐前缀计分、实际右补齐、极端±1000间隔、参考参数与源文件不变、状态reload，以及手算梯度。

## 产物与结果

忽略目录`build/experiments/24-preference-alignment/`包含冻结`sft-used.pt`、3个策略checkpoint及`results.json`。checkpoint沿用schema1，protocol记录22来源SHA、原21状态链、数据/模板指纹和本章训练预算。`figures.py`只读JSON重建6幅纯SVG，不再训练。二进制和运行日志由读者重建。

本次测试：SFT原始候选排序50%、DPO β1为58.3%而自由生成0/12；β0.1排序66.7%、生成2/12；翻转标签排序25%、生成0/12。相对参考变化排序分别50%（平局）/83.3%/83.3%/25%，不能冒充生成正确率。格式定义为一个g..l字母+LF，β0.1仅3/12符合；所有失败/字节保留。

二候选条件KL只比较子集内归一化概率，不是完整策略KL；参数负载141,312B不是进程峰值。固定参考SFT文件SHA256：

```text
77d94287f5b8d4f1df3a1aaa2c937ec026c1bf1e42c9c9380985faee61e75201
```

非法生成字节的替代解码字符保留；Windows控制台使用UTF8，不删除失败文本。相同本机环境验证的结果不保证跨版本逐位相同。
