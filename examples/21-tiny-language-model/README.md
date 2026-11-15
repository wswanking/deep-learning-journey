# 第21章实践：CPU微型字节语言模型

在仓库根目录执行：

```powershell
& ./.venv/Scripts/python.exe examples/21-tiny-language-model/run.py --write-figures
```

默认CPU、float32、1线程，无外部数据或模型下载。依赖项目`.venv`中的PyTorch与NumPy。完整解释见[章节正文](../../docs/chapters/21-tiny-language-model.md)。

## 文件与接口

- `corpus.jsonl`：48篇原创中英小短文，24个颜色/形状组；按组固定16/4/4划分。
- `corpus.py`：可编辑语料生成源。生成器不是训练时必须执行步骤，随仓库文本为实际输入。
- `tokenizer.py`：UTF8字节0..255，encode/decode，无特殊BOS/EOS/PAD。
- `model.py`：TinyLMConfig、TinyDecoderLM；forward(input_ids,attention_mask=None)返回[B,T,256]。mask True表示真实键，只支持非空真实前缀+右补齐；q_proj/v_proj独立，供后续LoRA复用。
- `run.py`：主模型、237步恢复对照、1层/少数据配置对照、一次主模型测试、固定6条生成与17检查。
- `figures.py`：从实际JSON生成6幅纯SVG；单独运行只重建图，不训练。

## 实验协议

主模型35,328参数，d32/heads4/layers2/FFN64/context64，pre-LN、GELU、dropout0。AdamW lr0.003、wd0.01，clip1，batch8，600步，307,200预测目标（有重复）。初始化seed20261115、独立窗口采样seed17；生成seed111、温度0.8或贪心。文档内随机窗口不跨文档。最终固定600步，不以验证选择检查点。

评估每文档按64预测目标切片，右侧输入0/mask False/target-100，累计NLL除真实目标总数。每段上下文重置；损失单位nat/byte，PPL=exp(CE)，不同tokenizer不可直接比较。测试在配置固定后仅主模型评估一次。

## 输出

输出位于忽略目录`build/experiments/21-tiny-language-model/`，不提交二进制产物：

- `results.json`：全部曲线、生成字节与有效UTF8标记、协议、版本、耗时、检查记录。
- `base-model.pt`：schema_version1，config、model_state、tokenizer、protocol。load_checkpoint返回(model,metadata)，供22/23/24只读复用。不是完整训练恢复文件。
- `step-237-checkpoint.pt`：另外含optimizer_state、next_step、采样/torch RNG、history；验证恢复至600与连续路线相同。

本次主模型train/validation CE0.867768/0.947393，test1.002630；1层validation0.713180，比2层更低；4文档组train0.688122但validation1.595341。全部事实保留，不宣称更深必然更好。17项检查通过；重复、颜色不一致、算术提示失败和一种非法UTF8采样均保留。资源数字只估参数与动量负载，不是进程峰值。

Windows控制台显式使用UTF8，保留非法序列解码的替代字符；不会删掉坏样例来避免打印错误。相同CPU版本下验证恢复相等，不保证跨硬件/版本逐位相同。
