# 全工程脚本平台清单

范围：全部仓库自有脚本；不含依赖、虚拟环境、Git内部及build生成文件。此清单不是仅检查扩展名：CI还会执行章节入口、图生成器及综合报告。

## 本轮补齐

- 第30章独立报告从硬编码Edge改为共享浏览器发现，支持Linux Chromium及显式环境变量。
- 第02/04章增加Linux进程峰值RSS采样，资源说明不再误标为Windows独有。
- 第11/14/20章静态绘图脚本改为按脚本位置定位仓库，避免依赖调用目录。
- README主构建区及全部30章实践README并列提供Linux命令，保留Windows入口。
- 第24章实跑暴露小批DPO训练震荡，统一使用完整36对训练批次与较小学习率；保留原22项检查，图生成不再混用历史数字。
- 新增audit-scripts与verify-experiments，隔离执行所有章节及生成器；CI保存逐步日志。

## 平台边界与验收证据

PowerShell环境检查的CIM、Windows注册表、Windows字体目录仅在Windows有意义，不应改名成sh后假称通用；对应Linux工具提供CPU/内存/磁盘/工具/字体/框架检查。在线GitHub页面检查需要网络，代码已接共享浏览器，未列入离线全实验验收。原始Windows硬件与框架JSON是历史证据，保留原路径，不改成Linux路径。

全量链路已由[Actions 44104896665](https://github.com/wswanking/deep-learning-journey/actions/runs/44104896665)实跑通过：55个步骤覆盖全部30章实验、9个静态生成器、10个独立绘图入口、综合报告导出/重放/质量门槛和重生成全书。每次后续提交仍重新执行整套CI，结果以对应提交为准。此轮详情见[Linux验收记录](appendices/linux-validation.md)。

## 逐文件清单（97项）

| 文件 | 类型与Linux处理 |
| --- | --- |
| `examples/01-linear-algebra/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/02-calculus/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/03-probability/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/04-numerical-optimization/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/04-numerical-optimization/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/05-machine-learning/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/06-perceptron/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/07-multilayer-perceptron/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/07-multilayer-perceptron/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/08-backpropagation/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/09-training-and-optimizers/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/09-training-and-optimizers/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/10-generalization/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/10-generalization/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/11-convolutional-networks/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/12-residual-and-normalization/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/12-residual-and-normalization/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/13-sequence-models/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/13-sequence-models/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/14-tokenization-and-embeddings/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/15-attention/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/15-attention/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/16-transformer/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/16-transformer/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/17-representation-learning/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/17-representation-learning/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/18-vae-and-gan/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/18-vae-and-gan/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/19-diffusion-models/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/19-diffusion-models/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/20-language-model-pretraining/corpus.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/20-language-model-pretraining/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/21-tiny-language-model/corpus.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/21-tiny-language-model/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/21-tiny-language-model/model.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/21-tiny-language-model/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/21-tiny-language-model/tokenizer.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/22-instruction-tuning/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/22-instruction-tuning/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/22-instruction-tuning/sft_data.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/23-parameter-efficient-tuning/adapters.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/23-parameter-efficient-tuning/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/23-parameter-efficient-tuning/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/24-preference-alignment/data.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/24-preference-alignment/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/24-preference-alignment/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/25-language-model-inference/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/25-language-model-inference/inference.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/25-language-model-inference/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/26-retrieval-augmented-generation/data.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/26-retrieval-augmented-generation/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/26-retrieval-augmented-generation/retrieval.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/26-retrieval-augmented-generation/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/27-model-evaluation/data.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/27-model-evaluation/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/27-model-evaluation/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/28-multimodal-representations/data.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/28-multimodal-representations/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/28-multimodal-representations/model.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/28-multimodal-representations/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/29-vision-language-models/data.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/29-vision-language-models/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/29-vision-language-models/prefix_model.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/29-vision-language-models/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `examples/30-capstone/figures.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/30-capstone/render-report.mjs` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `examples/30-capstone/report_text.py` | Python辅助模块；章节调用或独立绘图入口覆盖 |
| `examples/30-capstone/run.py` | 跨平台Python；全实验按章节顺序实跑 |
| `scripts/audit-scripts.py` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `scripts/browser.mjs` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `scripts/build.mjs` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `scripts/build.ps1` | Windows原生；对应Bash/Linux入口保留并列 |
| `scripts/build.sh` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `scripts/check-environment.ps1` | Windows原生；对应Bash/Linux入口保留并列 |
| `scripts/check-environment.py` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `scripts/check-environment.sh` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `scripts/check-github-math.mjs` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `scripts/check-markdown.mjs` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `scripts/check-publication.mjs` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `scripts/cpu-smoke.py` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `scripts/generate-chapter01-figures.mjs` | Node跨平台；在隔离副本实跑 |
| `scripts/generate-chapter02-figures.mjs` | Node跨平台；在隔离副本实跑 |
| `scripts/generate-chapter03-figures.mjs` | Node跨平台；在隔离副本实跑 |
| `scripts/generate-chapter05-figures.mjs` | Node跨平台；在隔离副本实跑 |
| `scripts/generate-chapter08-figures.mjs` | Node跨平台；在隔离副本实跑 |
| `scripts/generate-chapter11-figures.mjs` | Node跨平台；在隔离副本实跑 |
| `scripts/generate-chapter14-figures.mjs` | Node跨平台；在隔离副本实跑 |
| `scripts/generate-chapter20-figures.mjs` | Node跨平台；在隔离副本实跑 |
| `scripts/generate-sample-figures.mjs` | Node跨平台；在隔离副本实跑 |
| `scripts/setup.sh` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `scripts/verify-experiments.py` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `scripts/verify-experiments.sh` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `scripts/verify.sh` | 跨平台工具/模块；语法检查并由对应入口覆盖 |
| `tests/build.test.mjs` | Node回归测试；Windows专属模拟保留 |
| `tests/github-math.test.mjs` | Node回归测试；Windows专属模拟保留 |
| `tests/linux.test.mjs` | Node回归测试；Windows专属模拟保留 |
| `tests/markdown.test.mjs` | Node回归测试；Windows专属模拟保留 |
