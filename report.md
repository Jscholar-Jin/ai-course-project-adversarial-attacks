# CIFAR-10 对抗攻击与防御实验报告

## 1. 实验目标

本实验围绕 CIFAR-10 图像分类任务，研究深度学习模型在对抗扰动下的脆弱性，以及常见防御方法在不同攻击条件下的有效性。实验目标包括以下六项。

1. 实现至少 3 种对抗攻击方法。
2. 测试攻击成功率 ASR，并分析扰动大小和迭代次数的影响。
3. 实现至少 2 种防御方法。
4. 完成“攻击 vs 防御”的军备竞赛实验。
5. 保留对抗样本可视化，并分析模型在哪些样本上更脆弱。
6. 研究迁移攻击是否成立。

## 2. 实验设置

### 2.1 数据集

- 数据集为 CIFAR-10。
- 图像尺寸为 `32 x 32`。
- 分类类别总数为 10。

### 2.2 模型

- `CNN`：自定义卷积网络，作为主要白盒攻击目标。
- `ResNet18`：适配 CIFAR-10 的 ResNet，用作迁移攻击的源模型。
- `cnn_adv`：在标准 CNN 基础上进行对抗训练得到的鲁棒模型。

### 2.3 攻击方法

- `FGSM`
- `PGD`
- `DeepFool`

### 2.4 防御方法

- 对抗训练
- JPEG 压缩预处理
- Feature Squeezing

### 2.5 关键超参数

- 默认扰动预算为 `eps = 8/255`。
- `PGD alpha = 2/255`
- `PGD steps = 10`
- `DeepFool steps = 20`
- `JPEG quality = 75`
- `Feature squeeze bit depth = 5`

### 2.6 运行环境

- 使用本地 GPU 环境。
- Python 解释器为 `E:\Anaconda\Anaconda\envs\PyTorch\python.exe`
- 已验证 `torch 2.7.1+cu118`
- `CUDA available = True`

## 3. 模型训练结果

首先给出干净测试集上的基线精度。

- `ResNet18`：`93.35%`
- 标准 `CNN`：`91.65%`
- 对抗训练后的 `CNN`：`81.27%`

结论很直接。对抗训练能够为后续鲁棒性提升提供基础，但会带来明显的干净精度损失，标准 CNN 与对抗训练 CNN 的差距约为 10 个百分点。

相关图像：

- [clean_acc.png](figures/clean_acc.png)

训练日志文件：

- [logs/resnet.csv](logs/resnet.csv)
- [logs/cnn.csv](logs/cnn.csv)
- [logs/cnn_adv.csv](logs/cnn_adv.csv)

## 4. 白盒攻击结果

### 4.1 固定 `eps = 8/255` 的攻击强度对比

在标准 `CNN` 上进行白盒攻击时，结果如下。

- `FGSM`：`Clean Acc = 91.65%`，`Adv Acc = 12.09%`，`ASR = 86.82%`
- `PGD`：`Clean Acc = 91.65%`，`Adv Acc = 0.61%`，`ASR = 99.33%`
- `DeepFool`：`Clean Acc = 91.65%`，`Adv Acc = 0.29%`，`ASR = 99.68%`

可以看出：

- 单步攻击 `FGSM` 已经足以大幅降低分类准确率。
- 迭代攻击 `PGD` 与 `DeepFool` 几乎可以完全破坏标准 CNN 的分类性能。
- 在该实验设置下，迭代攻击明显强于单步攻击。

相关图像：

- [whitebox.png](figures/whitebox.png)

### 4.2 扰动大小 `eps` 的影响

当 `eps` 增大时，攻击成功率提升、对抗准确率下降，这一趋势在 `FGSM` 与 `PGD` 上都成立。

`FGSM` 的结果如下。

- `2/255`：`Adv Acc = 32.41%`，`ASR = 64.64%`
- `4/255`：`Adv Acc = 16.31%`，`ASR = 82.20%`
- `8/255`：`Adv Acc = 12.09%`，`ASR = 86.82%`
- `12/255`：`Adv Acc = 11.32%`，`ASR = 87.71%`

`PGD` 的结果如下。

- `2/255`：`Adv Acc = 14.36%`，`ASR = 84.33%`
- `4/255`：`Adv Acc = 1.86%`，`ASR = 97.97%`
- `8/255`：`Adv Acc = 0.61%`，`ASR = 99.33%`
- `12/255`：`Adv Acc = 0.20%`，`ASR = 99.78%`

结论如下。

- 随着扰动预算扩大，攻击能力持续增强。
- `PGD` 对 `eps` 更敏感，在 `4/255` 后几乎已经接近完全成功。
- `FGSM` 的上升也很明显，但整体仍弱于 `PGD`。

相关图像：

- [white_eps.png](figures/white_eps.png)

### 4.3 迭代次数 `steps` 的影响

固定 `eps = 8/255`，考察 `PGD` 迭代次数的影响，结果如下。

- `steps = 1`：`Adv Acc = 24.47%`，`ASR = 73.30%`
- `steps = 3`：`Adv Acc = 3.96%`，`ASR = 95.68%`
- `steps = 5`：`Adv Acc = 1.70%`，`ASR = 98.15%`
- `steps = 10`：`Adv Acc = 0.60%`，`ASR = 99.35%`

可以得到两个结论。

- 增加迭代次数会稳定增强攻击效果。
- 从 `1` 步增加到 `3` 步的提升最明显，之后收益逐渐趋于饱和。

相关图像：

- [pgd_steps.png](figures/pgd_steps.png)

## 5. 迁移攻击结果

在迁移攻击实验中，对抗样本由 `ResNet18` 生成，再攻击目标 `CNN`。结果如下。

- `FGSM`：`Clean Acc = 91.65%`，`Adv Acc = 37.22%`，`ASR = 59.77%`
- `PGD`：`Clean Acc = 91.65%`，`Adv Acc = 16.90%`，`ASR = 81.57%`
- `DeepFool`：`Clean Acc = 91.65%`，`Adv Acc = 89.09%`，`ASR = 2.82%`

结果说明：

- 迁移攻击是成立的，尤其 `PGD` 仍然保持较强攻击能力。
- `FGSM` 也具备一定迁移性，但弱于 `PGD`。
- `DeepFool` 的迁移性很差，说明它更贴合源模型的局部决策边界，跨模型泛化较弱。

相关图像：

- [transfer.png](figures/transfer.png)

## 6. 防御结果

### 6.1 对抗训练

在对抗训练模型 `cnn_adv` 上重新进行攻击，结果如下。

- `FGSM`：`Clean Acc = 81.27%`，`Adv Acc = 41.59%`，`ASR = 48.82%`
- `PGD`：`Clean Acc = 81.27%`，`Adv Acc = 33.75%`，`ASR = 58.47%`
- `DeepFool`：`Clean Acc = 81.27%`，`Adv Acc = 48.74%`，`ASR = 40.03%`

与标准 CNN 相比，可以观察到：

- 对抗训练显著提高了鲁棒性。
- 在 `PGD` 攻击下，对抗准确率从 `0.61%` 提升到 `33.75%`。
- 这种提升的代价是干净精度从 `91.65%` 降到 `81.27%`。

相关图像：

- [adv_train.png](figures/adv_train.png)

### 6.2 输入预处理防御

JPEG 压缩防御结果如下。

- `FGSM`：`Clean Acc = 83.94%`，`Adv Acc = 27.67%`，`ASR = 67.31%`
- `PGD`：`Clean Acc = 83.94%`，`Adv Acc = 18.89%`，`ASR = 77.53%`
- `DeepFool`：`Clean Acc = 83.94%`，`Adv Acc = 79.26%`，`ASR = 5.96%`

Feature Squeezing 防御结果如下。

- `FGSM`：`Clean Acc = 91.48%`，`Adv Acc = 12.30%`，`ASR = 86.58%`
- `PGD`：`Clean Acc = 91.48%`，`Adv Acc = 0.56%`，`ASR = 99.39%`
- `DeepFool`：`Clean Acc = 91.48%`，`Adv Acc = 73.16%`，`ASR = 20.03%`

这一部分可以总结为：

- JPEG 对 `DeepFool` 的抑制效果较明显。
- Feature Squeezing 对 `DeepFool` 也有一定缓解作用。
- 但对 `FGSM` 和尤其 `PGD`，两种预处理防御都不稳定，整体效果弱于对抗训练。

相关图像：

- [preprocess.png](figures/preprocess.png)

## 7. 自适应攻击与军备竞赛

为了检验预处理防御是否真的鲁棒，进一步采用 `BPDA + PGD` 自适应攻击。结果如下。

- `JPEG + Adaptive PGD`：`Clean Acc = 83.94%`，`Adv Acc = 2.45%`，`ASR = 97.08%`
- `Feature Squeeze + Adaptive PGD`：`Clean Acc = 91.48%`，`Adv Acc = 0.54%`，`ASR = 99.41%`

这一部分是本实验里最关键的防御分析之一。

- 在普通攻击下，预处理方法看起来有时有效。
- 一旦攻击者把防御机制纳入攻击图中，攻击成功率迅速回到极高水平。
- 这说明 JPEG 压缩与特征压缩更多是在制造梯度不连续或梯度遮蔽，而不是真正改变模型的鲁棒边界。
- 相比之下，对抗训练更接近真实鲁棒性。

相关图像：

- [adaptive.png](figures/adaptive.png)

## 8. 可视化与脆弱样本分析

### 8.1 对抗样本可视化

已经生成以下可视化结果。

- [vis_examples.png](figures/vis_examples.png)
- [vis_tsne.png](figures/vis_tsne.png)

其中：

- `vis_examples.png` 在一张图中同时展示 `FGSM`、`PGD`、`DeepFool` 的原图、对抗样本和放大扰动。
- `vis_tsne.png` 展示正常样本与三种对抗样本在特征空间中的分布偏移。

从图中可以直接看出两个现象。

- 扰动在视觉上通常难以察觉，但足以显著改变模型预测。
- 对抗样本在特征空间中与正常样本明显分离，说明攻击不只是改变最终输出，而是影响了中间表征。

### 8.2 脆弱类别分析

脆弱类别统计来自可视化子集。按照成功攻击样本数量排序，最脆弱的类别如下。

- `frog`：`fooled_count = 51`，`mean_margin_drop = 12.27`
- `ship`：`fooled_count = 46`，`mean_margin_drop = 16.32`
- `cat`：`fooled_count = 41`，`mean_margin_drop = 5.97`
- `horse`：`fooled_count = 39`，`mean_margin_drop = 14.80`
- `airplane`：`fooled_count = 39`，`mean_margin_drop = 9.77`

相关图像与数据文件：

- [vis_fragile.png](figures/vis_fragile.png)
- [vis_fragile.csv](figures/vis_fragile.csv)

这一部分说明：

- `frog`、`ship`、`cat` 等类别在当前可视化子集中更容易被攻击成功。
- 堆叠柱状图还能看出三种攻击都对这些类别有贡献，不是某一种攻击单独造成的异常。

## 9. 总结

本实验可以归纳出以下结论。

1. 标准 CNN 对对抗扰动非常敏感，`PGD` 与 `DeepFool` 几乎可以完全破坏其分类能力。
2. 扰动预算 `eps` 与迭代次数 `steps` 增大时，攻击效果显著增强。
3. 迁移攻击是成立的，其中 `PGD` 展现出更强的跨模型迁移性。
4. 对抗训练是更有效、更稳定的鲁棒防御方法，但会降低干净样本精度。
5. JPEG 压缩和特征压缩在非自适应评估中可能表现出一定效果，但在自适应攻击下会明显失效。
6. 对抗样本可视化与 t-SNE 结果表明，对抗攻击会显著改变模型的特征表示。

## 10. 关键结果文件

- [results/metrics.csv](results/metrics.csv)
- [results/summary.csv](results/summary.csv)
- [README.md](README.md)
