## VAE：

主要参考资料：苏剑林：变分自编码器(一)（二）

VAE本质上就是AE引入了高斯噪声作为中间变量，从而能够产生生成的效果。

VAE主要是想利用一个简单的分布$p(z)$，$p(z)$服从与高斯分布。从$p(z)$采样的点能够生成原始的分布$p(x)$，也就是得到$q(x|z)$这个这个条件分布。

联合概率密度 $p(x,z)$ 捕捉了数据 $x$ 和潜在变量 $z$ 的整个生成过程。这意味着我们考虑了从潜在空间到数据空间的所有可能路径以及它们发生的概率。

通过 $p(x,z)$ 我们可以直接讨论和建模潜在变量 $z$ 如何生成数据 $x$ 的问题，同时也能反向推断出给定数据 $x$ 时潜在变量 $z$ 的分布。这涵盖了整个模型的编码（推断）和解码（生成）过程。

## LOSS推导

所以可以从联合概率分布开始推导，我们从真实的联合概率密度分$p(x,z)$和近似的联合概率密度分布$q(x,z)$的KL散度入手。用$q(x,z)$来近似$p(x,z)$

从而优化这个KL散度。对应KL散度，我们进行变化，可以让其得到KL散度为$\mathbb{E}_{p(x)}\left[ \int p(z|x) \ln \left( \frac{\tilde{p}(x)p(z|x)}{q(x, z)} \right) dz \right]$，继续将其拆解可以发现其中的$\mathbb{E}_{x\sim p(x)} \left[ \int p(z|x) \ln \tilde{p}(x) p(z|x) \, dz \right] = \mathbb{E}_{x\sim p(x)} \left[ \ln \tilde{p}(x) \int p(z|x) \, dz \right] = \mathbb{E}_{x\sim p(x)} \left[ \ln \tilde{p}(x) \right]$是一个常数

因此我们可以得到**我们所需要的损失**$L= KL散度-常数=\mathbb{E}_{x \sim p(x)}\left[ \int p(z|x) \ln \left( \frac{p(z|x)}{q(x, z)} \right) dz \right] = \mathbb{E}_{x \sim p(x)} \left[ \mathbb{E}_{z \sim p(z|x)} \left[ - \ln q(x|z) \right] + KL\left( p(z|x) \parallel q(z) \right) \right]$

这时可以看见L由两个部分组成，左边为$q(x|z)$ 也就是decoder的部分，右边为$p(z|x)$也需要进一步的近似，得到$q(z|x)$为encoder的部分。

可以看到 当我们优化这个$L$​的时候两部分的Loss会相互抵抗，达成一个平衡效果。

在这个$L$中，我们对$q(x|z),p(z|x),q(z)$​未知

## Encoder部分

如果我们假设$z$服从高斯分布，那么$q(z)$就知道了

对于$p(z|x)$仍然需要近似，近似$p(z|x)$的方法同理与近似$p(x,z)$的方法，都是用KL散度进行近似，可以得到一个

$\hat{p}(z|x) = q(z|x) = \frac{q(z|x)q(z)}{q(x)} = \frac{q(z|x)q(z)}{\int q(z|x)q(z)dz}$，但是分母上的积分是不可能的事情

所以我们直接假设$p(z|x)$服从正态分布，均值和方差由神经网络得到$p(z|x) = \frac{1}{\prod_{k=1}^{D} \sqrt{2\pi\sigma^2_{(k)}(x)}} \exp\left(-\frac{1}{2} \left\| \frac{z - \mu(x)}{\sigma(x)} \right\|^2\right)$​

其中d是分量的维度，并且之所以指数函数不需要对各分量做累积的原因是，指数内部的范数平方已经包含了累积的步骤$\left\| \frac{z - \mu(x)}{\sigma(x)} \right\|^2 = \sum_{k=1}^{d} \left( \frac{z_k - \mu_k(x)}{\sigma_{k}(x)} \right)^2$​

那么最终的得到的**Encoder的部分的损失**$KL\left( p(z|x) \parallel q(z) \right) = \frac{1}{2} \sum_{k=1}^{d} \left( \mu_k^2(x) + \sigma_k^2(x) - \ln \sigma_k^2(x) - 1 \right)$

## Decoder部分

对于Decoder，我们可以直接同样用认为$q(x|z)$服从高斯分布

这样$q(x|z) = \frac{1}{\prod_{k=1}^{D} \sqrt{2\pi\sigma^2_{(k)}(z)}} \exp\left(-\frac{1}{2} \left\| \frac{x - \tilde{\mu}(z)}{\tilde{\sigma}(z)} \right\|^2\right)$

$-\ln q(x|z) = \frac{1}{2} \left\| \frac{x - \tilde{\mu}(z)}{\tilde{\sigma}(z)} \right\|^2 + \frac{D}{2} \ln 2\pi + \frac{1}{2} \sum_{k=1}^{D} \ln \tilde{\sigma}^2_{(k)}(z)$

这里的$\tilde{\sigma}(z)和\tilde{\mu}(z)$都是decoder网络计算出的参数

如果我们假设$\tilde{\sigma}$为常数的话，那么可以接着等价于优化以下的公式

$-\ln q(x|z) \sim \frac{1}{2\tilde{\sigma}^2} \left\| x - \tilde{\mu}(z) \right\|^2$

## 采样技巧

在VAE中我们神经网络中的一个batch的每一个$x$都从$p(z|x)$中采样一个专属于这个$x$的$z$，然后接着用这个$z$去计算$-\ln{q(x|z)}$。由于只采样了一个样本，因此我们内部的期望可以去掉。最终得到的损失函数如下：
$L=\mathbb{E}_{x \sim p(x)} \left[ - \ln q(x|z)  + KL\left( p(z|x) \parallel q(z) \right) \right]$