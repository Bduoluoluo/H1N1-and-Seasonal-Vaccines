### (1)

逻辑回归模型为

$$
P(y=1\mid x)=\frac{1}{1+e^{-x}}=p
$$

$$
P(y=0\mid x)=\frac{e^{-x}}{1+e^{-x}}=1-p
$$

假设有 $N$ 个训练样本 $x_1,x_2,\cdots ,x_N$，训练标签为 $y_1,y_2,\cdots ,y_N$

所以对于训练样本 $(x_i,y_i)$ 的输出概率为

$$
P(y_i)=p_i^{y_i}(1-p_i)^{1-y_i}
$$

我们认为所有训练样本相互独立，因此得到的目标函数为

$$
F(w)=\prod_{i=1}^{N}{p_i^{y_i}(1-p_i)^{1-y_i}}
$$

将目标函数取对数，得到

$$
\ln{F(w)}=\sum_{i=1}^{N}{y_i\ln{p_i}+(1-y_i)\ln{(1-p_i)}}
$$

$$
=\sum_{i=1}^{N}{y_i\ln{\frac{e^x}{e^x+1}}+(1-y_i)\ln{\frac{1}{e^x+1}}}
$$

$$
=\sum_{i=1}^{N}{y_ix-y_i\ln{(e^x+1)}-(1-y_i)\ln{(e^x+1)}}
$$
 
$$
=\sum_{i=1}^{N}{y_ix-\ln{(e^x+1)}}
$$

其中 $x=w^Tx_i$。

对其求导得到

$$
\frac{\partial \ln{F(w)}}{\partial w}=\sum_{i=1}^{N}{y_ix_i-\frac{x_ie^{x}}{e^x+1}}
$$

$$
=\sum_{i=1}^{N}{(y_i-\frac{e^{x}}{e^x+1})x_i}
$$

$$
=\sum_{i=1}^{N}{(y_i-p_i)x_i}
$$

### (2)

标签取值更改为 $\{1,-1\}$ 后，训练样本 $(x_i,y_i)$ 的输出概率为

$$
P(y_i)=\sqrt{p_i^{1+y_i}(1-p_i)^{1-y_i}}
$$

所以其目标函数为

$$
F(w)=\prod_{i=1}^{N}{\sqrt{p_i^{1+y_i}(1-p_i)^{1-y_i}}}
$$

将目标函数取对数得到

$$
\ln{F(w)}=\frac{1}{2}\sum_{i=1}^{N}{(1+y_i)\ln{p_i}+(1-y_i)\ln{(1-p_i)}}
$$

$$
=\frac{1}{2}\sum_{i=1}^{N}{\ln{\frac{e^x}{e^x+1}}+y_ix-\ln{(e^x+1)}}
$$

$$
=\frac{1}{2}\sum_{i=1}^{N}{x+y_ix-2\ln{(e^x+1)}}
$$

对其求导得到

$$
\frac{\partial \ln{F(w)}}{\partial w}=\frac{1}{2}\sum_{i=1}^{N}{x_i+y_ix_i-2\frac{x_ie^{x}}{e^x+1}}
$$

$$
=\frac{1}{2}\sum_{i=1}^{N}{(1+y_i-2\frac{e^{x}}{e^x+1})x_i}
$$

### (3)

设原目标函数为 $F_0$

#### L1 正则化

目标函数修改为

$$
F=F_0+\frac{\lambda}{N}|w|
$$

对目标函数求导得到

$$
\frac{\partial F}{\partial w}=\frac{\partial F_0}{\partial w}+\frac{\lambda}{N}\text{sgn}(w)
$$

因此如果 $w$ 是正数，则梯度下降会让 $w$ 变小；如果 $w$ 是负数，则梯度下降会让 $w$ 变大。即 $w$ 会趋向于 $0$。

#### L2 正则化

目标函数修改为
$$
F=F_0+\frac{\lambda}{2N}|w|^2
$$

对目标函数求导得到
$$
\frac{\partial F}{\partial w}=\frac{\partial F_0}{\partial w}+\frac{\lambda}{N}w
$$

类似的，如果 $w$ 是正数，则梯度下降会让 $w$ 变小；如果 $w$ 是负数，则梯度下降会让 $w$ 变大。不过 L1 正则化中变化量是固定的，而在 L2 正则化中若 $|w|$ 越大，则变化量也越大。

### (4)

$$
F(\mathbf{x}) = \text{sgn}(\sum_{i=1}^{N}\alpha_i y_i k(\mathbf{x}_i, \mathbf{x}) + b)
$$
