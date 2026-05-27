# ai-course-project-adversarial-attacks
# Adversarial Attack Transferability

本项目是人工智能课程大作业，主要研究图像分类模型中的对抗样本攻击与迁移攻击问题。实验基于 PyTorch 实现，比较了 FGSM、PGD、MI-FGSM 等典型攻击方法在白盒攻击和跨模型迁移攻击下的攻击成功率。

## 1. 项目简介

随着深度神经网络在图像分类任务中的广泛应用，模型对微小扰动的鲁棒性问题受到关注。对抗样本通过在人眼几乎不可察觉的范围内添加扰动，可以导致模型预测错误。

本项目围绕以下问题展开：

- 白盒攻击下，不同攻击方法对目标模型的影响；
- 迁移攻击下，源模型生成的对抗样本能否攻击目标模型；
- FGSM、PGD、MI-FGSM 在攻击成功率和扰动效果上的差异。

## 2. 实验设置

### 数据集

- CIFAR-10

### 模型

- Model A: ResNet18
- Model B: SimpleCNN

### 攻击方法

- FGSM
- PGD
- MI-FGSM

### 扰动强度

- epsilon = 8 / 255

## 3. 项目结构

```text
models/      模型定义
attacks/     对抗攻击算法
scripts/     训练、攻击和评估脚本
utils/       工具函数
results/     实验结果
configs/     配置文件

## 4.Python 环境配置

本项目基于 Python 和 PyTorch 实现，建议使用 Conda 创建独立环境运行。

### 1. 创建 Conda 环境

```bash
conda create -n adv_attack python=3.9
conda activate adv_attack


pip install -r requirements.txt

