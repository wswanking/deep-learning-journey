# 第29章CPU实践：连续视觉前缀与三任务问答

[完整教程](../../docs/chapters/29-vision-language-models.md)

## 运行与依赖

在仓库根目录，先按21章与28章入口生成各自基座：

~~~powershell
& ./.venv/Scripts/python.exe -B examples/29-vision-language-models/run.py
~~~

Linux / WSL：

~~~bash
.venv-linux/bin/python -B examples/29-vision-language-models/run.py
~~~

只用项目Python3.11.5、NumPy2.4.6、PyTorch2.15.0+cpu；CPU float32单线程，无额外下载。只读加载21/28模型、28原创图像生成器，不改它们的代码/检查点。全局同名model.py/data.py通过独立模块名加载，避免导入冲突。

默认输出build/experiments/29-vision-language-models/results.json、connector-only.pt、joint.pt、text-prior.pt及三个记录JSONL；图像由确定生成器复现，不保存外部图片或大数据缓存。

## 实际模型与数据

- 冻结28视觉token路径，输出[B,16,32]；视觉总11216参数含未使用图像级投影528，token路径10688。
- 连接器共享MLP32→64→32，4192参数；21语言模型35328参数，context64。
- 连续视觉前缀占16位置，再接文字embedding，组合长度≤64；从本章wrapper读取21内部模块，不修改21源码。
- 新渲染seed2901/2902/2903，9已见概念：3颜色×3形状。训练/验证/测试288/72/108图，每图color/shape/describe三问，864/216/324条。
- 先按图分集合，检查精确像素hash与28全部图像无交集。同9类仍共享，不是未知概念、自然图像或新模板泛化。
- 问题只有任务名+冒号+LF，答案为ASCII属性文字+LF。模型输入没有class_id/真实属性文字。
- loss只在回答字节计算，视觉/问题/padding目标−100；首回答位于P+prompt_bytes−1。LF只作本任务停止约定。

## 两阶段和对照

阶段1只用describe，冻结视觉与语言，4192参连接器训练400步/lr.01；按describe验证NLL选最佳。阶段2从相同阶段1best出发，联合连接器+语言共39520参、三任务800步/lr.002。追加恒定全黑RGB输入的训练对照，模型/起点/问题/答案/采样顺序/预算相同，只去掉图像区分信息。

统一AdamW，betas(.9,.999)、eps1e-8、decay0、batch16，初始化seed2900、采样seed2904，50步验证选最低有效回答byte平均NLL。冻结后才测试与干预；贪心、最多20新byte、LF停止。

2026-11-17首轮34项通过，71.258秒含导入2.607秒；复跑25.778秒。耗时受并发负载/缓存影响，非稳定吞吐。特征缓存958464B；未测RSS/GPU显存。

| 方法 | 最佳步 | 测试回答NLL | 完整正确/324 | color/108 | shape/108 | describe/108 |
| --- | --- | --- | --- | --- | --- | --- |
| 连接器only | 400 | 2.189331 | 12 | 0 | 0 | 12 |
| 联合微调 | 750 | 0.004191 | 321 | 108 | 106 | 107 |
| 恒定黑图训练 | 600 | 0.192343 | 84 | 36 | 36 | 12 |

表中测试NLL统一含三任务，阶段1选择只看describe（验证0.918273），范围不同。联合/黑图模型格式都324/324；黑图teacher-forced字节准确率88.57%仍不代表看图。恒定输入每问题只能给固定答案，均衡下完整上限(1/3+1/3+1/9)/3=7/27，实测84/324恰好达到。

换图class+4 mod9（颜色/形状都变）后，联合模型按原答案2/324、按新图321/324；遮全黑图按原答案0/324。遮图没有原任务类别真值，不能称统一幻觉率。保留3条正常输入错误，未按测试调参。单种子小任务不代表开放视觉语言能力。

## 可复用API

~~~python
from prefix_model import (
    TinyPrefixVLM, load_vlm_checkpoint, vision_api
)
model, metadata = load_vlm_checkpoint(
    "build/experiments/29-vision-language-models/joint.pt",
    "build/experiments/21-tiny-language-model/base-model.pt",
    "build/experiments/28-multimodal-representations/dual-encoder.pt",
)
dual, _ = vision_api.load_checkpoint(
    "build/experiments/28-multimodal-representations/dual-encoder.pt"
)
features = dual.vision.eval().forward_tokens(images)  # B,16,32
logits = model.forward_features(features, ids, mask)  # B,16+T,256
~~~

代码需按本章目录导入或动态导入prefix_model。模型输出含视觉位置，生成首字节取最后有效问题位置的logit；不能直接按T−1忽略P偏移。完整批生成可复用run.generate_records(model,records,feature_bank)，records由data.make_data生成。三问题提示和答案定义在data.py。

模型文件保存语言/连接器全状态、准确21/28文件SHA、配置/词表/阶段/数据协议；不包含视觉权重。加载严格检查两个源检查点hash。属于推理文件，未保存优化器/采样状态续训。

## 34项核验与复现

核验图像组/上游精确像素隔离、视觉/MLP尺寸、线性手算及NumPy/梯度、6样本首回答对齐、视觉忽略标签、零前缀退化为共享LM、未来因果干预、float32/float64 padding、组合上下文拒绝、回答CE手算/忽略局部梯度、冻结LM仍能反向训练连接器、阶段参数数/有限梯度/重载、冻结状态与原文件不变、同批次序列、恒定图同问题输出一致。

float32 padding有效logit差4.53e-6、容差1e-5；同权重float64交叉检查7.99e-15、容差1e-11。三模型保存恢复逐元素一致。

~~~powershell
& ./.venv/Scripts/python.exe -B examples/29-vision-language-models/run.py --output-dir build/experiments/29-vision-language-models/repeated --skip-figures
& ./scripts/build.ps1 -Format all -Chapter 29
node scripts/check-publication.mjs --name chapter-29 --capture
~~~

Linux / WSL：

~~~bash
.venv-linux/bin/python -B examples/29-vision-language-models/run.py --output-dir build/experiments/29-vision-language-models/repeated --skip-figures
bash scripts/build.sh --format all --chapter 29
node scripts/check-publication.mjs --name chapter-29 --capture
~~~

跨平台允许声明的数值误差，不要求耗时相同。重复比较包含全部轨迹/选择/生成/干预及模型张量；两次34项检查及全部轨迹、生成和模型张量一致。HTML/PDF16页/49公式/6SVG/36链接，1440/390与PDF审计通过，16页/6图已逐页及放大检查。

## 文件与SVG

prefix_model.py定义连续前缀wrapper与检查点；data.py负责独立渲染/分组/回答移位；run.py负责训练/评价/干预；figures.py生成6幅真正SVG。前3图原创教学，后3图来自结果。实际RGB输入作为256个矢量色格表达，不嵌入位图。LLaVA/BLIP2/POPE仅为核验主来源，未运行它们的大模型或标准基准。


## 最终出版验收

2026-11-17两次34项CPU核验通过，复跑25.778秒。数据/环境/协议、全部训练与验证轨迹、最佳步、逐例生成、干预结果及三个模型的全部张量逐项一致；原21/28检查点指纹不变，冻结视觉和阶段1语言状态一致，三模型推理重载逐元素相同。HTML/PDF实际生成，16页、49处公式、6幅SVG、36链接，1440/390px无页面溢出、断锚、数学解析错误、未渲染Markdown或图中文字越界；PDF零位图，字体全部嵌入、含书签/标签，无纸张外文字。已渲染16页逐页查看，并放大核对6图、手算公式、表格、代码与分页；图中区分按原答案和按当前图评分。RSS/GPU显存未测，范围为已见9概念的新扰动，保留正常图3条完整错误和冻结语言阶段的不足；最终状态与提交由总计划维护。
