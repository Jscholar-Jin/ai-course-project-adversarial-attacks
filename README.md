# CIFAR-10 对抗攻击与防御实验

本项目完成以下内容：

- 4 种攻击：`FGSM`、`PGD`、`DeepFool` 、`自适应BPDA` 
- 2 种防御：对抗训练、输入预处理（`JPEG` 压缩、特征压缩）
- 攻击成功率 `ASR` 测试
- 扰动大小 `eps` 和迭代次数 `steps` 影响分析
- 攻击 vs 防御军备竞赛
- 迁移攻击：`resnet -> cnn`
- 可视化：对抗样本、放大扰动、t-SNE、脆弱类别统计

## 文件说明

```text
models.py                       模型定义，包括 SimpleCNN、CIFAR-10 适配版 ResNet18，以及模型加载接口
data_utils.py                   CIFAR-10 数据加载、数据预处理、DataLoader 构建与随机种子设置
attacks.py                      对抗攻击核心实现，包括 FGSM、PGD、MI-FGSM、DeepFool、SPSA 和统一攻击接口 make_adv
defenses.py                     防御方法实现，包括 JPEG 压缩、位深压缩、输入预处理封装和 BPDA 近似反向传播

train.py                        通用训练脚本，支持标准训练和基于攻击样本的对抗训练
train_adv.py                    FGSM 对抗训练脚本，训练并保存 FGSM 对抗训练后的鲁棒模型
train_adv_final.py              PGD 对抗训练最终版脚本，训练更强的 PGD 对抗训练模型

eval_attacks.py                 攻击效果评估脚本，评估白盒攻击、迁移攻击和黑盒攻击的 Adv Acc 与 ASR
eval.py                         通用评估脚本，可统一评估攻击、防御、迁移攻击和不同目标模型表现
eval_adv.py                     对抗训练模型评估脚本，评估 FGSM/PGD 对抗训练模型在多种攻击下的鲁棒性
eval_bpda_adaptive.py           BPDA 自适应攻击评估脚本，用于检测 JPEG 压缩和位深压缩是否存在梯度遮蔽

eps.py                          扰动大小 eps 敏感性实验，分析不同 eps 下攻击成功率和对抗样本准确率变化
steps.py                        迭代次数 steps 敏感性实验，分析 Transfer DeepFool 在不同 steps 下的迁移攻击效果

summary.py                      实验结果汇总脚本，生成 Clean Acc、攻击对比、防御对比、自适应攻击等统计图和汇总表
tsne.py                         t-SNE 特征分布可视化脚本，用于分析 Clean 样本与对抗样本在特征空间中的分布差异
visualize_v3.py                 攻击样本可视化脚本，生成 Clean、FGSM、PGD、DeepFool 四行对比图
visual_adv_diff_and_fragile.py  对抗样本差异与脆弱样本分析脚本，生成扰动放大图、脆弱类别统计和脆弱样本表

export_word.py                  报告导出脚本，用于将 Markdown 报告内容转换为 Word 文档
report.md                       实验报告 Markdown 版本，记录实验目标、方法、结果与分析
README.md                       GitHub 项目说明文档，介绍项目功能、运行方式和实验结果
requirements.txt                Python 依赖列表，用于配置运行环境
```

## 输出文件

训练日志：

- logs/cnn.csv                         SimpleCNN baseline 训练日志
- logs/resnet.csv                      ResNet18 baseline 训练日志
- logs/fgsm_adv_training_log.csv       FGSM 对抗训练日志
- logs/pgd_adv_training_log.csv        PGD 对抗训练日志

模型权重：

- checkpoints/cnn.pt                   SimpleCNN baseline 模型权重
- checkpoints/resnet.pt                ResNet18 baseline 模型权重
- checkpoints/cnn_fgsm_adv_train.pt    FGSM 对抗训练模型权重
- checkpoints/cnn_pgd_adv_final.pt     PGD 对抗训练最终模型权重

结果表：
- results/metrics.csv                          baseline 模型训练指标
- results/attack_effect_results.csv            白盒、迁移、黑盒攻击实验结果
- results/eps_sensitivity.csv                  扰动大小 eps 敏感性实验结果
- results/steps_sensitivity.csv                迭代次数 steps 敏感性实验结果
- results/transfer_deepfool_steps_sensitivity.csv  Transfer DeepFool steps 敏感性实验结果
- results/adv_training_eval_results.csv        对抗训练模型鲁棒性评估结果
- results/bpda_adaptive_results.csv            BPDA 自适应攻击评估结果
- results/tsne_features.csv                    t-SNE 特征降维数据
- results/fragile_class_stats.csv              最脆弱类别统计结果
- results/fragile_samples.csv                  最脆弱样本统计结果

图像与可视化：

- figures/all_attack_effects.png               全部攻击方法效果总览图
- figures/attack_adv_acc_bar.png               不同攻击方法 Adv Acc 对比柱状图
- figures/attack_asr_bar.png                   不同攻击方法 ASR 对比柱状图

- figures/eps_sensitivity_adv_acc.png          eps 对 Adv Acc 的影响曲线
- figures/eps_sensitivity_asr.png              eps 对 ASR 的影响曲线
- figures/steps_sensitivity_adv_acc.png        steps 对 Adv Acc 的影响曲线
- figures/steps_sensitivity_asr.png            steps 对 ASR 的影响曲线

- figures/transfer_deepfool_steps_adv_acc.png  Transfer DeepFool steps 对 Adv Acc 的影响曲线
- figures/transfer_deepfool_steps_asr.png      Transfer DeepFool steps 对 ASR 的影响曲线

- figures/adv_training_adv_acc_bar.png         对抗训练模型 Adv Acc 对比图
- figures/adv_training_asr_bar.png             对抗训练模型 ASR 对比图

- figures/bpda_adaptive_adv_acc_bar.png        BPDA 自适应攻击 Adv Acc 对比图
- figures/bpda_adaptive_asr_bar.png            BPDA 自适应攻击 ASR 对比图

- figures/vis_examples.png                     对抗样本示例可视化图
- figures/vis_4rows.png                        Clean、FGSM、PGD、DeepFool 四行对比图
- figures/vis_fgsm.png                         FGSM 攻击样本可视化图
- figures/vis_pgd.png                          PGD 攻击样本可视化图
- figures/vis_deepfool.png                     DeepFool 攻击样本可视化图

- figures/tsne_clean_vs_attacks.png            Clean 与多种对抗样本 t-SNE 总体分布图
- figures/tsne_clean_vs_each_attack.png        Clean 与不同攻击样本 t-SNE 对比图
- figures/tsne_attack_success_selected.png     攻击成功样本 t-SNE 分布图
- figures/vis_tsne.png                         t-SNE 可视化汇总图

- figures/adv_diff_examples.png                原图、放大扰动与对抗样本可视化图
- figures/fragile_class_counts.png             脆弱类别攻击成功次数统计图
- figures/fragile_class_asr.png                脆弱类别平均 ASR 统计图
- figures/vis_fragile.png                      脆弱类别可视化汇总图
- figures/vis_fragile.csv                      脆弱样本与类别可视化数据

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
