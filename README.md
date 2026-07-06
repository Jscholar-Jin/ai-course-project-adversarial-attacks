# 基于 CIFAR-10 的对抗样本迁移攻击与鲁棒防御实验

本项目基于 CIFAR-10 数据集，围绕图像分类模型的对抗样本攻击与鲁棒防御展开实验。项目实现了白盒攻击、迁移攻击、黑盒攻击、输入预处理防御、对抗训练防御、BPDA 自适应攻击，以及对抗样本可视化、t-SNE 特征分布分析和脆弱类别统计。

## 项目简介

本实验主要研究深度学习图像分类模型在对抗扰动下的鲁棒性问题。首先训练两个 baseline 分类模型：

- `Model A`：ResNet18，作为迁移攻击源模型；
- `Model B`：SimpleCNN，作为主要被攻击目标模型。

随后在 CIFAR-10 测试集上分别进行：

- 白盒攻击：`Model B -> Model B`
- 迁移攻击：`Model A -> Model B`
- 黑盒攻击：`Query Model B -> Model B`

攻击方法包括：

- FGSM
- PGD
- MI-FGSM
- DeepFool
- SPSA

防御方法包括：

- JPEG 压缩
- 位深压缩 / Feature Squeezing
- FGSM 对抗训练
- PGD 对抗训练
- BPDA 自适应攻击评估

---

## 项目结构

```text
AI-Adversarial-CIFAR10/
├── models.py
├── data_utils.py
├── attacks.py
├── defenses.py
├── train.py
├── train_adv.py
├── train_adv_final.py
├── eval.py
├── eval_attacks.py
├── eval_adv.py
├── eval_bpda_adaptive.py
├── eps.py
├── steps.py
├── summary.py
├── tsne.py
├── visualize_v3.py
├── visual_adv_diff_and_fragile.py
├── requirements.txt
├── README.md
├── checkpoints/
├── logs/
├── results/
└── figures/
```

---

## 文件说明

```text
models.py                       模型定义，包括 SimpleCNN、CIFAR-10 适配版 ResNet18，以及模型加载接口
data_utils.py                   CIFAR-10 数据加载、数据预处理、DataLoader 构建与随机种子设置
attacks.py                      对抗攻击核心实现，包括 FGSM、PGD、MI-FGSM、DeepFool、SPSA 和统一攻击接口 make_adv
defenses.py                     防御方法实现，包括 JPEG 压缩、位深压缩、输入预处理封装和 BPDA 近似反向传播

train.py                        通用训练脚本，用于训练 SimpleCNN 和 ResNet18 baseline 模型
train_adv.py                    FGSM 对抗训练脚本，训练并保存 FGSM 对抗训练后的鲁棒模型
train_adv_final.py              PGD 对抗训练最终版脚本，训练更强的 PGD 对抗训练模型

eval.py                         通用评估脚本，可统一评估攻击、防御、迁移攻击和不同目标模型表现
eval_attacks.py                 攻击效果评估脚本，评估白盒攻击、迁移攻击和黑盒攻击的 Adv Acc 与 ASR
eval_adv.py                     对抗训练模型评估脚本，评估 FGSM / PGD 对抗训练模型在多种攻击下的鲁棒性
eval_bpda_adaptive.py           BPDA 自适应攻击评估脚本，用于检测 JPEG 压缩和位深压缩是否存在梯度遮蔽

eps.py                          扰动大小 eps 敏感性实验，分析不同 eps 下攻击成功率和对抗样本准确率变化
steps.py                        迭代次数 steps 敏感性实验，分析不同 steps 下 PGD、MI-FGSM、DeepFool 等攻击效果变化

summary.py                      实验结果汇总脚本，生成 Clean Acc、攻击对比、防御对比、自适应攻击等统计图和汇总表
tsne.py                         t-SNE 特征分布可视化脚本，用于分析 Clean 样本与对抗样本在特征空间中的分布差异
visualize_v3.py                 攻击样本可视化脚本，生成 Clean、FGSM、PGD、DeepFool 四行对比图
visual_adv_diff_and_fragile.py  对抗样本差异与脆弱样本分析脚本，生成扰动放大图、脆弱类别统计和脆弱样本表

requirements.txt                Python 依赖列表，用于配置运行环境
README.md                       GitHub 项目说明文档
```

---

## 环境配置

本机 GPU 环境可直接使用：

```powershell
$PY = "E:\Anaconda\Anaconda\envs\PyTorch\python.exe"
```

已验证环境：

```text
torch 2.7.1+cu118
CUDA available = True
```

也可以直接写成：

```powershell
& "E:\Anaconda\Anaconda\envs\PyTorch\python.exe" xxx.py
```

安装依赖：

```powershell
pip install -r requirements.txt
```

---

## 当前实验设置

```text
数据集：CIFAR-10

源模型 Model A：ResNet18
目标模型 Model B：SimpleCNN

白盒攻击路径：Model B -> Model B
迁移攻击路径：Model A -> Model B
黑盒攻击路径：Query Model B -> Model B

默认攻击预算：eps = 8/255
PGD alpha：2/255
PGD steps：10
DeepFool steps：20
SPSA samples：16
SPSA max samples：1000

JPEG quality：75
Feature squeeze bit depth：5

FGSM 对抗训练：
clean weight = 0.5
adv weight = 0.5

PGD 对抗训练：
train PGD steps = 7
eval PGD steps = 10
```

---

## 运行命令

### 1. 训练标准 baseline 模型

训练 `SimpleCNN` 和 `ResNet18` 两个标准模型：

```powershell
& $PY train.py --model both --epochs 30 --batch-size 128 --workers 2 --download
```

输出文件：

```text
checkpoints/cnn.pt
checkpoints/resnet.pt
logs/cnn.csv
logs/resnet.csv
```

---

### 2. FGSM 对抗训练

```powershell
& $PY train_adv.py --model cnn --clean-checkpoint ./checkpoints/cnn.pt --save-path ./checkpoints/cnn_fgsm_adv_train.pt --log-path ./logs/fgsm_adv_training_log.csv --epochs 10 --batch-size 128 --workers 2 --eps 0.0313725 --clean-weight 0.5 --adv-weight 0.5
```

输出文件：

```text
checkpoints/cnn_fgsm_adv_train.pt
logs/fgsm_adv_training_log.csv
```

---

### 3. PGD 对抗训练最终版

```powershell
& $PY train_adv_final.py --model cnn --init-checkpoint ./checkpoints/cnn.pt --save-path ./checkpoints/cnn_pgd_adv_final.pt --log-path ./logs/pgd_adv_training_log.csv --epochs 20 --batch-size 128 --workers 2 --eps 0.0313725 --alpha 0.0078431 --train-pgd-steps 7 --eval-pgd-steps 10 --clean-weight 0.5 --adv-weight 0.5
```

输出文件：

```text
checkpoints/cnn_pgd_adv_final.pt
logs/pgd_adv_training_log.csv
```

---

### 4. 评估白盒、迁移和黑盒攻击效果

```powershell
& $PY eval_attacks.py --batch-size 128 --workers 2 --model-a resnet --checkpoint-a ./checkpoints/resnet.pt --model-b cnn --checkpoint-b ./checkpoints/cnn.pt --eps 0.0313725 --alpha 0.0078431 --steps 10 --deepfool-steps 20 --spsa-samples 16 --spsa-max-samples 1000
```

输出文件：

```text
results/attack_effect_results.csv
figures/attack_adv_acc_bar.png
figures/attack_asr_bar.png
figures/all_attack_effects.png
```

---

### 5. 扰动大小 eps 敏感性实验

```powershell
& $PY eps.py --batch-size 128 --workers 2 --model-a resnet --checkpoint-a ./checkpoints/resnet.pt --model-b cnn --checkpoint-b ./checkpoints/cnn.pt --eps-list 2,4,8,12,16 --alpha 0.0078431 --steps 10 --deepfool-steps 20
```

输出文件：

```text
results/eps_sensitivity.csv
figures/eps_sensitivity_asr.png
figures/eps_sensitivity_adv_acc.png
```

---

### 6. 迭代次数 steps 敏感性实验

```powershell
& $PY steps.py --batch-size 128 --workers 2 --model-a resnet --checkpoint-a ./checkpoints/resnet.pt --model-b cnn --checkpoint-b ./checkpoints/cnn.pt --steps-list 1,3,5,10,20 --eps 0.0313725 --alpha 0.0078431
```

输出文件：

```text
results/steps_sensitivity.csv
figures/steps_sensitivity_asr.png
figures/steps_sensitivity_adv_acc.png
results/transfer_deepfool_steps_sensitivity.csv
figures/transfer_deepfool_steps_asr.png
figures/transfer_deepfool_steps_adv_acc.png
```

---

### 7. 评估 FGSM 对抗训练模型

```powershell
& $PY eval_adv.py --batch-size 128 --workers 2 --model-a resnet --checkpoint-a ./checkpoints/resnet.pt --model-b cnn --checkpoint-b-adv ./checkpoints/cnn_fgsm_adv_train.pt --eps 0.0313725 --alpha 0.0078431 --steps 10 --deepfool-steps 20 --spsa-samples 16 --spsa-max-samples 1000
```

输出文件：

```text
results/adv_training_eval_results.csv
figures/adv_training_adv_acc_bar.png
figures/adv_training_asr_bar.png
```

---

### 8. 评估 PGD 对抗训练模型

```powershell
& $PY eval_adv.py --batch-size 128 --workers 2 --model-a resnet --checkpoint-a ./checkpoints/resnet.pt --model-b cnn --checkpoint-b-adv ./checkpoints/cnn_pgd_adv_final.pt --eps 0.0313725 --alpha 0.0078431 --steps 10 --deepfool-steps 20 --spsa-samples 16 --spsa-max-samples 1000
```

输出文件：

```text
results/adv_training_eval_results.csv
figures/adv_training_adv_acc_bar.png
figures/adv_training_asr_bar.png
```

---

### 9. BPDA 自适应攻击评估

```powershell
& $PY eval_bpda_adaptive.py --model cnn --checkpoint ./checkpoints/cnn.pt --defenses jpeg,squeeze --jpeg-quality 75 --bit-depth 5 --batch-size 64 --workers 2 --eps 0.0313725 --alpha 0.0078431 --steps 10
```

输出文件：

```text
results/bpda_adaptive_results.csv
figures/bpda_adaptive_adv_acc_bar.png
figures/bpda_adaptive_asr_bar.png
```

---

### 10. 生成汇总图

```powershell
& $PY summary.py
```

输出文件：

```text
figures/clean_acc.png
figures/whitebox.png
figures/white_eps.png
figures/pgd_steps.png
figures/transfer.png
figures/adv_train.png
figures/preprocess.png
figures/adaptive.png
results/summary.csv
```

---

### 11. 生成攻击样本四行可视化图

```powershell
& $PY visualize_v3.py --model cnn --checkpoint ./checkpoints/cnn.pt --batch-size 128 --workers 2 --eps 0.0313725 --alpha 0.0078431 --steps 10 --deepfool-steps 20 --num-examples 4 --max-samples 1000 --output-name vis_4rows.png
```

输出文件：

```text
figures/vis_4rows.png
```

---

### 12. 生成 t-SNE 特征分布图

```powershell
& $PY tsne.py --model cnn --checkpoint ./checkpoints/cnn.pt --attacks fgsm,pgd,deepfool --num-samples 300 --batch-size 64 --workers 2 --eps 0.0313725 --alpha 0.0078431 --steps 10 --deepfool-steps 20 --prefix tsne
```

输出文件：

```text
results/tsne_features.csv
figures/tsne_clean_vs_attacks.png
figures/tsne_clean_vs_each_attack.png
figures/tsne_attack_success_selected.png
```

---

### 13. 生成扰动放大图与脆弱类别分析

```powershell
& $PY visual_adv_diff_and_fragile.py --model cnn --checkpoint ./checkpoints/cnn.pt --attacks fgsm,pgd,deepfool --batch-size 128 --workers 2 --eps 0.0313725 --alpha 0.0078431 --steps 10 --deepfool-steps 20 --num-examples-per-attack 3 --magnify 12 --prefix fragile
```

输出文件：

```text
results/fragile_class_stats.csv
results/fragile_samples.csv
figures/adv_diff_examples.png
figures/fragile_class_counts.png
figures/fragile_class_asr.png
```

---

## 输出文件说明

### 训练日志

```text
logs/cnn.csv                         SimpleCNN baseline 训练日志
logs/resnet.csv                      ResNet18 baseline 训练日志
logs/fgsm_adv_training_log.csv       FGSM 对抗训练日志
logs/pgd_adv_training_log.csv        PGD 对抗训练日志
```

### 模型权重

```text
checkpoints/cnn.pt                   SimpleCNN baseline 模型权重
checkpoints/resnet.pt                ResNet18 baseline 模型权重
checkpoints/cnn_fgsm_adv_train.pt    FGSM 对抗训练模型权重
checkpoints/cnn_pgd_adv_final.pt     PGD 对抗训练最终模型权重
```

### 结果表

```text
results/metrics.csv                          通用评估指标表
results/summary.csv                          汇总后的摘要表
results/attack_effect_results.csv            白盒、迁移、黑盒攻击实验结果
results/eps_sensitivity.csv                  eps 敏感性实验结果
results/steps_sensitivity.csv                steps 敏感性实验结果
results/transfer_deepfool_steps_sensitivity.csv  Transfer DeepFool steps 敏感性实验结果
results/adv_training_eval_results.csv        对抗训练模型鲁棒性评估结果
results/bpda_adaptive_results.csv            BPDA 自适应攻击评估结果
results/tsne_features.csv                    t-SNE 特征降维数据
results/fragile_class_stats.csv              最脆弱类别统计结果
results/fragile_samples.csv                  最脆弱样本统计结果
```

### 图像与可视化

```text
figures/clean_acc.png                         干净样本准确率与鲁棒性代价图
figures/whitebox.png                          白盒攻击效果图
figures/white_eps.png                         白盒攻击 eps 影响图
figures/pgd_steps.png                         PGD steps 影响图
figures/transfer.png                          迁移攻击效果图
figures/adv_train.png                         对抗训练防御效果图
figures/preprocess.png                        输入预处理防御效果图
figures/adaptive.png                          BPDA 自适应攻击效果图

figures/all_attack_effects.png                全部攻击方法效果总览图
figures/attack_adv_acc_bar.png                不同攻击方法 Adv Acc 对比柱状图
figures/attack_asr_bar.png                    不同攻击方法 ASR 对比柱状图

figures/eps_sensitivity_adv_acc.png           eps 对 Adv Acc 的影响曲线
figures/eps_sensitivity_asr.png               eps 对 ASR 的影响曲线
figures/steps_sensitivity_adv_acc.png         steps 对 Adv Acc 的影响曲线
figures/steps_sensitivity_asr.png             steps 对 ASR 的影响曲线

figures/transfer_deepfool_steps_adv_acc.png   Transfer DeepFool steps 对 Adv Acc 的影响曲线
figures/transfer_deepfool_steps_asr.png       Transfer DeepFool steps 对 ASR 的影响曲线

figures/adv_training_adv_acc_bar.png          对抗训练模型 Adv Acc 对比图
figures/adv_training_asr_bar.png              对抗训练模型 ASR 对比图

figures/bpda_adaptive_adv_acc_bar.png         BPDA 自适应攻击 Adv Acc 对比图
figures/bpda_adaptive_asr_bar.png             BPDA 自适应攻击 ASR 对比图

figures/vis_examples.png                      FGSM、PGD、DeepFool 原图、对抗图、扰动图综合可视化
figures/vis_4rows.png                         Clean、FGSM、PGD、DeepFool 四行对比图
figures/vis_fgsm.png                          FGSM 攻击样本可视化图
figures/vis_pgd.png                           PGD 攻击样本可视化图
figures/vis_deepfool.png                      DeepFool 攻击样本可视化图

figures/vis_tsne.png                          t-SNE 可视化汇总图
figures/tsne_clean_vs_attacks.png             Clean 与多种对抗样本 t-SNE 总体分布图
figures/tsne_clean_vs_each_attack.png         Clean 与不同攻击样本 t-SNE 对比图
figures/tsne_attack_success_selected.png      攻击成功样本 t-SNE 分布图

figures/adv_diff_examples.png                 原图、放大扰动与对抗样本可视化图
figures/fragile_class_counts.png              脆弱类别攻击成功次数统计图
figures/fragile_class_asr.png                 脆弱类别平均 ASR 统计图
figures/vis_fragile.png                       脆弱类别可视化汇总图
figures/vis_fragile.csv                       脆弱样本与类别可视化数据
```

---

## 模型权重说明

由于模型权重文件较大，未直接上传至 GitHub。可通过训练脚本重新生成：

```powershell
& $PY train.py --model both --epochs 30 --batch-size 128 --workers 2 --download
& $PY train_adv.py --model cnn --clean-checkpoint ./checkpoints/cnn.pt --save-path ./checkpoints/cnn_fgsm_adv_train.pt --log-path ./logs/fgsm_adv_training_log.csv --epochs 10 --batch-size 128 --workers 2 --eps 0.0313725 --clean-weight 0.5 --adv-weight 0.5
& $PY train_adv_final.py --model cnn --init-checkpoint ./checkpoints/cnn.pt --save-path ./checkpoints/cnn_pgd_adv_final.pt --log-path ./logs/pgd_adv_training_log.csv --epochs 20 --batch-size 128 --workers 2 --eps 0.0313725 --alpha 0.0078431 --train-pgd-steps 7 --eval-pgd-steps 10 --clean-weight 0.5 --adv-weight 0.5
```

训练完成后会生成：

```text
checkpoints/cnn.pt
checkpoints/resnet.pt
checkpoints/cnn_fgsm_adv_train.pt
checkpoints/cnn_pgd_adv_final.pt
```

建议在 `.gitignore` 中加入：

```gitignore
checkpoints/
*.pt
*.pth
data/
__pycache__/
*.pyc
```

---

## 实验结论

```text
1. 白盒攻击中 PGD 攻击能力最强，普通模型在强白盒攻击下几乎被完全攻破。
2. 迁移攻击中 MI-FGSM 表现最好，动量机制能够增强对抗样本跨模型迁移能力。
3. 输入预处理防御实现简单，但面对 PGD 和 BPDA 自适应攻击时防御效果有限。
4. FGSM 对抗训练能提升对单步攻击的防御能力，但难以抵抗强多步 PGD 攻击。
5. PGD 对抗训练综合鲁棒性最好，在白盒攻击、迁移攻击和黑盒 SPSA 下表现更稳定。
6. t-SNE 和扰动放大图表明，对抗扰动虽然在像素空间中较小，但会显著改变模型特征分布。
```
