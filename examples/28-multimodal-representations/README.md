# 第28章：原创图文双塔CPU训练

```powershell
& ./.venv/Scripts/python.exe -B examples/28-multimodal-representations/run.py
```

从仓库根目录运行，无网络、无第三方图片、无下载权重。默认1200步；改 `--updates` 后不能沿用正文数字。

## 数据与接口

`data.py` 原创RGB生成器：红/绿/蓝×圆/方/十字，16×16图像、随机中心/尺寸/亮度/背景/噪声。train/validation/test独立种子2801/2802/2803，每类64/16/24图，总576/144/216；逐像素哈希无重复，但九种概念全部在训练出现，不是未见类别测试。数据和项目同许可。

文字输入为颜色索引和形状索引，各0～2；展示中文由固定词表生成，模型不接受任意中文句子，也不复用21章UTF-8词元。

`model.py` 提供 `TinyDualEncoder(DualEncoderConfig())`。`.vision.forward_tokens(images)` 接收RGB float32 `[B,3,16,16]`、值域0～1，输出 `[B,16,32]`；`.vision(images)` 与 `.text(attributes long[B,2])` 输出归一化 `[B,16]`。视觉11216参数、全模型11840。一个4头PreLN视觉Transformer，patch4/宽32/前馈64、dropout0；平均池化和16维投影，非原始CLIP完整复现。

## 训练与评价

初始化种子2800、批次种子2804，CPU单线程float32、确定性算法。AdamW lr0.001、weight decay0.001、betas(0.9,0.999)、epsilon1e-8；batch9，每概念一图避免假负例。双向对角交叉熵，温度0.1固定。每50步及第1步验证图→文九类平均NLL，最低点1050恢复；测试只在配置和模型冻结后评价。

2026-11-15真实结果：测试图→文Hit@1=206/216=95.37%，Hit@3=216/216；文→图每概念24张图都计相关，Hit@1=9/9。十个图→文错误均在同颜色内混淆形状，JSON保留全部预测/混淆矩阵。随机起点22/216；故意循环错配候选文字却保留旧标签4/216，属于评价映射负对照，不是额外训练。

手算loss0.31326168751822286与自动微分相符；重复语义假负例损失ln2；patch卷积/展开线性误差1.79e-7，保存加载视觉输出逐位相等。训练与检查3.270秒、含导入6.054秒；数据张量2882880字节，未测峰值进程内存。环境Python3.11.5、NumPy2.4.6、PyTorch2.15.0+cpu。

## 产物与复用

- `build/experiments/28-multimodal-representations/results.json`：曲线、全部检索预测、数据指纹、来源协议、原始测试代表图像数值。
- 同目录 `dual-encoder.pt`：schema1、config、model_state、protocol；`load_checkpoint(path)` 返回模型和元数据。
- `figures.py`：从实际JSON生成6幅真正SVG，图像像素画成独立矢量矩形，无嵌入位图。

最终检查点SHA256：

```text
da3c49769b23599a817c651d4d49bbdeaac65ee592d68a69f1727e9e07a293a7
```

后续章节可以只读复制冻结视觉编码器并调用 `forward_tokens`；需要记录此SHA和原协议。该文件不含优化器完整状态，验证的是推理恢复，未做中断续训。训练只覆盖简单合成图形与固定属性，不支持自然照片、任意文字、未见类别或开放世界检索结论。
