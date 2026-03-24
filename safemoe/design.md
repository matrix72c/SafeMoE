### 核心目标

在一个已经具备双重知识（安全知识 + 有害知识）的预训练 MoE 模型上，通过引入新增的“危害专家 (Harmful Experts)”并结合路由/梯度掩码机制进行持续预训练，实现：

1. 富集 (Enrichment)：将有害知识从原有专家中剥离，并集中吸入到新增的 Harmful Experts 中。
2. 剔除 (Ablation)：在推理阶段安全地切除 Harmful Experts，实现有害能力的根除，同时保持通用能力。

### Step 1. PT阶段

目的：验证SGTM算法在MoE上可以实现 Std Experts 只保存通用知识，Harmful Experts只保存有害知识。

模型参数分为三部分：$\theta_{harmful}$包括Harmful Attention Heads和Harmful Experts；$\theta_{std}$包括剩余的Attention Heads和Std Experts；$\theta_{shared}$包括剩余的参数，如路由、Embedding。

- 选择性梯度掩码 (针对 $D_{harmful}$)：在反向传播计算完所有梯度后，SGTM 会将 $θ_{std}$ 的梯度强制置为零（即 $\nabla \theta = \{\nabla \theta_{harmful}, 0\}$。这确保了遗忘领域的样本只会更新专属的 $θ_{harmful}$ 参数，从而防止目标知识的信息流入保留参数中。
- 选择性参数掩码 (针对 $D_{std}$)：在前向传播阶段，当输入为保留数据时，SGTM 会将 $θ_{harmful}$ 参数暂时掩码置零（其对应的激活值也会变为零）。这强制模型学会在缺失 $θ_{harmful}$ 参数的情况下，依然能依靠 $θ_{std}$ 在保留数据集上表现良好。
- 常规训练（针对 $D_{unlabeled}$）：未标注数据不应用任何掩码，执行常规的前向和反向传播。
- $\theta_{shared}$在三类数据上均正常更新
- 能力移除 (Ablation)：训练完全结束后，直接将专属参数永久置零（$θ_{harmful} = 0$），从而“切除”目标知识，同时保证模型其他知识大部分完好无损。

### 实验设置

在英语/西班牙语双语的 Tiny Stories 数据集上，(100 - x)%的西班牙语是危害知识$D_{harmful}$，25\%的英语是通用知识$D_{std}$，75%的英语和x%的西班牙语是无标签数据 $D_{unlabeled}$。

### Step 2. CPT阶段

#### Model initialization

随机挑选 k 个 experts 和 n 个 Attention head，将它们的参数和 Experts 路由对应的列复制并添加微小噪声，作为新增的 $\theta_{harmful}$，原有 Experts 和 Attention head 参数作为 $\theta_{std}$

#### Warm up

数据：$D_{harmful}$ 和 $D_{std}$ 混合

因为 Std Experts 本身也懂有害知识，我们不能仅仅依赖语言模型的 NTP loss。我们需要在warm up阶段引入一个强制性的路由损失函数，鼓励 Router 会把 $D_{harmful}$ 引导给 Harmful Experts，从而让 Harmful Experts 在 $D_{harmful}$ 上能力更强

令 $z_t = \sum_{i \in H} P_i(x_t)$，$y_t=1$ 表示 harmful token，$y_t=0$ 表示 std token，$τ_h$ 可设 0.7 左右，$τ_s$ 可设 0.05~0.1
$$L_{route} =
\lambda_h  \mathbb{E}_{t: y_t=1} [\max(0, \tau_h - z_t)]^2
+
\lambda_s  \mathbb{E}_{t: y_t=0} [\max(0, z_t - \tau_s)]^2$$

验证：无标签的 $D_{harmful}$ 会自然流向 Harmful Experts，$D_{std}$ 流向 Std Experts

#### Knowledge transfer

数据：$D_{unlabeled}$与部分 $D_{harmful}$ 和 $D_{std}$ 混合

使用SGTM范式训练，通过大量 unlabled 的通用知识数据，将危害知识从 Std Experts 中“挤”到 Harmful Experts 中

验证：对抗微调成本显著上升