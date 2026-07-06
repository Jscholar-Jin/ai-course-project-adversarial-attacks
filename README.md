# CIFAR-10 对抗攻击与防御实验

本项目完成以下内容：

- 3 种攻击：`FGSM`、`PGD`、`DeepFool`
- 2 种防御：对抗训练、输入预处理（`JPEG` 压缩、特征压缩）
- 攻击成功率 `ASR` 测试
- 扰动大小 `eps` 和迭代次数 `steps` 影响分析
- 攻击 vs 防御军备竞赛
- 迁移攻击：`resnet -> cnn`
- 可视化：对抗样本、放大扰动、t-SNE、脆弱类别统计

## 文件说明

```text
models.py      模型定义
data_utils.py  数据加载与随机种子
attacks.py     FGSM / PGD / DeepFool
defenses.py    JPEG / 特征压缩 / BPDA
train.py       标准训练与对抗训练
eval.py        攻击、防御、迁移、自适应评估
summary.py     汇总表和统计图
visualize.py   样本可视化、扰动图、t-SNE
```

## 输出文件

训练日志：

- `logs/cnn.csv`
- `logs/resnet.csv`
- `logs/cnn_adv.csv`

模型权重：

- `checkpoints/cnn.pt`
- `checkpoints/resnet.pt`
- `checkpoints/cnn_adv.pt`

结果表：

- `results/metrics.csv`
- `results/summary.csv`

图像与可视化：

- `figures/clean_acc.png`
- `figures/whitebox.png`
- `figures/white_eps.png`
- `figures/pgd_steps.png`
- `figures/transfer.png`
- `figures/adv_train.png`
- `figures/preprocess.png`
- `figures/adaptive.png`
- `figures/vis_examples.png`
- `figures/vis_tsne.png`
- `figures/vis_fragile.png`
- `figures/vis_fragile.csv`

## 环境

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
