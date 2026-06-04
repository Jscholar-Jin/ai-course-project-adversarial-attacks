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

本机 GPU 环境可直接用：

```powershell
& 'E:\Anaconda\Anaconda\envs\PyTorch\python.exe'
```

已验证：

- `torch 2.7.1+cu118`
- `CUDA available = True`

## 运行命令

1. 训练标准模型：

```powershell
& 'E:\Anaconda\Anaconda\envs\PyTorch\python.exe' train.py --model both --epochs 30 --batch-size 128 --workers 2 --download
```

2. 训练对抗训练模型：

```powershell
& 'E:\Anaconda\Anaconda\envs\PyTorch\python.exe' train.py --model cnn --training adv --attack pgd --epochs 15 --batch-size 128 --workers 2 --eps 0.0313725 --alpha 0.0078431 --steps 7 --lr 0.01 --init-checkpoint ./checkpoints/cnn.pt --clean-weight 0.5 --adv-weight 0.5
```

3. 运行评估：

```powershell
& 'E:\Anaconda\Anaconda\envs\PyTorch\python.exe' eval.py --batch-size 128 --workers 2 --eps-list 2/255,4/255,8/255,12/255 --step-list 1,3,5,10 --jpeg-quality 75 --bit-depth 5 --output metrics.csv
```

4. 生成汇总图：

```powershell
& 'E:\Anaconda\Anaconda\envs\PyTorch\python.exe' summary.py
```

5. 生成可视化：

```powershell
& 'E:\Anaconda\Anaconda\envs\PyTorch\python.exe' visualize.py --model cnn --checkpoint ./checkpoints/cnn.pt --batch-size 128 --workers 2 --max-samples 384 --tsne-samples 384 --num-examples 4 --prefix vis
```

## 当前实验设置

- 数据集：`CIFAR-10`
- 白盒模型：`cnn`
- 迁移源模型：`resnet`
- 默认攻击预算：`eps = 8/255`
- `PGD alpha = 2/255`
- `PGD steps = 10`
- `DeepFool steps = 20`
- `JPEG quality = 75`
- `Feature squeeze bit depth = 5`

## 说明

- `metrics.csv` 是完整实验表，适合后续写报告分析。
- `summary.csv` 是整理后的摘要表。
- `clean_acc.png` 展示干净样本精度与鲁棒性代价。
- `whitebox.png`、`white_eps.png`、`pgd_steps.png` 分别展示白盒攻击强度、`eps` 影响和迭代次数影响。
- `transfer.png`、`adv_train.png`、`preprocess.png`、`adaptive.png` 分别对应迁移攻击、防御对比、预处理防御、自适应攻击。
- `vis_examples.png` 在一张图里同时展示 `FGSM`、`PGD`、`DeepFool` 的原图、对抗图、放大扰动。
- `vis_tsne.png` 用于展示正常样本与对抗样本特征分布差异。
- `vis_fragile.png` 和 `vis_fragile.csv` 给出可视化子集上的脆弱类别统计。
