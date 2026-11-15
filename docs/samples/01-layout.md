# 中文、公式与结构图

> 排版样张第一部分。这里使用小型数学例子检验显示效果，不代替第01章或第07章的正式讲解。

## 1. 阅读目标与中文层级

本页检查中文标点、粗体、行距、标题层级与中英文混排。核心概念首次出现应同时写明中文和英文，例如仿射变换（affine transformation）。正式教程还需补足动机、推导、手算、实验与练习。

正文中的 **输入、参数与输出** 应能在图和公式之间逐一对应。变量使用标准数学符号；正文解释与图示相邻，避免只看公式才能理解步骤。

## 2. 矢量结构图

![图 1：仿射变换的数据流与张量形状；本项目原创 SVG 示意图。](../assets/typesetting/linear-flow.svg)

图中采用列向量约定：输入有两个分量，输出有三个分量。方框、箭头、文字均为矢量对象，没有截图或嵌入位图。

## 3. 数学符号与维度

| 符号 | 含义 | 形状 |
| --- | --- | --- |
| $\mathbf{x}$ | 单个输入样本 | $2\times1$ |
| $\mathbf{W}$ | 权重矩阵 | $3\times2$ |
| $\mathbf{b}$ | 偏置向量 | $3\times1$ |
| $\mathbf{z}$ | 输出向量 | $3\times1$ |

单样本的仿射变换为：

$$
\mathbf{z}=\mathbf{W}\mathbf{x}+\mathbf{b},
\qquad
\mathbf{W}\in\mathbb{R}^{3\times2},
\quad
\mathbf{x}\in\mathbb{R}^{2},
\quad
\mathbf{b}\in\mathbb{R}^{3}.
$$

用一组小数字检查矩阵和多行公式：

$$
\begin{aligned}
\mathbf{W}&=\begin{bmatrix}1&2\\0&1\\-1&1\end{bmatrix},
&
\mathbf{x}&=\begin{bmatrix}2\\1\end{bmatrix},
&
\mathbf{b}&=\begin{bmatrix}0\\1\\0\end{bmatrix},\\
\mathbf{z}&=
\begin{bmatrix}4\\1\\-1\end{bmatrix}
+
\begin{bmatrix}0\\1\\0\end{bmatrix}
=
\begin{bmatrix}4\\2\\-1\end{bmatrix}.
\end{aligned}
$$

每一行由矩阵的一行与输入向量做内积，再加上对应偏置。后续正式章节会逐步解释这些运算；这里重点验证列向量、矩阵括号、上标、下标和多行对齐都能正确显示。

## 4. 代码、长行与表格

以下示例只使用 Python 标准库，变量对应上面的手算：

```python
weights = [[1.0, 2.0], [0.0, 1.0], [-1.0, 1.0]]
inputs = [2.0, 1.0]
bias = [0.0, 1.0, 0.0]
outputs = [sum(w * x for w, x in zip(row, inputs)) + offset for row, offset in zip(weights, bias)]
print(outputs)  # [4.0, 2.0, -1.0]
```

长代码行需要自动换行，不应超出纸张。表格里的中文与短公式也应有足够空间。

| 检查对象 | HTML 预期 | PDF 预期 |
| --- | --- | --- |
| 中文与数学混排 | 正文字体与数学字体协调 | 中文可检索且无缺字 |
| 较长代码行 | 自动换行、保留高亮 | 不裁切，允许合理跨页 |
| SVG 结构图 | 可连续缩放 | 保留路径与文字 |
| 章节导航 | 点击跳转 | 目录、链接与书签可用 |

## 5. 跨文档导航

下一部分是[矢量曲线、热图与跨章导航](02-vectors.md)。构建为一本样张时，此链接应跳转至同一输出中的对应章节；单独构建时，未选入的本地源文件作为附件保留。

线性代数的进一步学习参考 [MIT 18.06 官方课程](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)。这里引用课程作为阅读入口，以上数值例子和图为本项目原创样张。
