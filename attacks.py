# attacks.py
# 功能：
# 实现 FGSM / PGD / MI-FGSM / DeepFool / SPSA 攻击
#
# 输入图像默认范围：[0, 1]
# 适用于 CIFAR-10 图像：[B, 3, 32, 32]

import torch
import torch.nn.functional as F


def clamp(x, min_value=0.0, max_value=1.0):
    return torch.clamp(x, min_value, max_value)


# =========================================================
# 1. FGSM Attack
# =========================================================
def fgsm_attack(model, images, labels, eps=8 / 255):
    """
    FGSM: Fast Gradient Sign Method

    公式：
    x_adv = x + eps * sign(grad_x loss)

    特点：
    单步攻击，速度快。
    """

    model.eval()

    images = images.clone().detach()
    labels = labels.clone().detach()

    images.requires_grad = True

    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)

    model.zero_grad(set_to_none=True)
    loss.backward()

    grad_sign = images.grad.sign()

    adv_images = images + eps * grad_sign
    adv_images = clamp(adv_images, 0.0, 1.0)

    return adv_images.detach()


# =========================================================
# 2. PGD Attack
# =========================================================
def pgd_attack(
    model,
    images,
    labels,
    eps=8 / 255,
    alpha=2 / 255,
    steps=10,
    random_start=True
):
    """
    PGD: Projected Gradient Descent

    公式：
    x_{t+1} = Proj_{B_eps(x)}(x_t + alpha * sign(grad_x loss))

    特点：
    多步迭代攻击，每一步之后都投影回 eps 范围。
    """

    model.eval()

    ori_images = images.clone().detach()
    labels = labels.clone().detach()

    if random_start:
        adv_images = ori_images + torch.empty_like(ori_images).uniform_(-eps, eps)
        adv_images = clamp(adv_images, 0.0, 1.0)
    else:
        adv_images = ori_images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True

        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad(set_to_none=True)
        loss.backward()

        grad_sign = adv_images.grad.sign()

        adv_images = adv_images + alpha * grad_sign

        delta = torch.clamp(
            adv_images - ori_images,
            min=-eps,
            max=eps
        )

        adv_images = clamp(ori_images + delta, 0.0, 1.0).detach()

    return adv_images.detach()


# =========================================================
# 3. MI-FGSM Attack
# =========================================================
def mi_fgsm_attack(
    model,
    images,
    labels,
    eps=8 / 255,
    alpha=2 / 255,
    steps=10,
    decay=1.0
):
    """
    MI-FGSM: Momentum Iterative FGSM

    思想：
    在迭代攻击中加入动量项，使更新方向更加稳定。

    公式：
    g_{t+1} = decay * g_t + grad / ||grad||_1
    x_{t+1} = x_t + alpha * sign(g_{t+1})

    特点：
    迁移攻击能力通常比普通 FGSM / PGD 更强。
    """

    model.eval()

    ori_images = images.clone().detach()
    labels = labels.clone().detach()

    adv_images = ori_images.clone().detach()
    momentum = torch.zeros_like(adv_images)

    for _ in range(steps):
        adv_images.requires_grad = True

        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad(set_to_none=True)
        loss.backward()

        grad = adv_images.grad.detach()

        # 按每张图像做 L1 归一化
        grad_norm = torch.mean(
            torch.abs(grad),
            dim=(1, 2, 3),
            keepdim=True
        )

        grad = grad / (grad_norm + 1e-8)

        momentum = decay * momentum + grad

        adv_images = adv_images + alpha * momentum.sign()

        delta = torch.clamp(
            adv_images - ori_images,
            min=-eps,
            max=eps
        )

        adv_images = clamp(ori_images + delta, 0.0, 1.0).detach()

    return adv_images.detach()


# =========================================================
# 4. DeepFool Attack
# =========================================================
def deepfool_single(
    model,
    image,
    eps=8 / 255,
    max_steps=20,
    overshoot=0.02,
    num_classes=10
):
    """
    单张图片的 DeepFool 攻击。

    DeepFool 思想：
    1. 找到当前预测类别；
    2. 对其他类别分别近似分类边界；
    3. 选择距离最近的边界；
    4. 沿最短方向移动；
    5. 重复直到分类改变。

    这里额外加入 eps 限制，方便和 FGSM / PGD 在相同扰动范围下对比。
    """

    model.eval()

    ori_image = image.clone().detach()
    adv_image = ori_image.clone().detach()

    with torch.no_grad():
        ori_output = model(ori_image)
        ori_pred = ori_output.argmax(dim=1).item()

    for _ in range(max_steps):
        adv_image.requires_grad = True

        output = model(adv_image)
        current_pred = output.argmax(dim=1).item()

        # 如果已经分类改变，攻击成功，停止
        if current_pred != ori_pred:
            adv_image = adv_image.detach()
            break

        logits = output[0]
        ori_logit = logits[ori_pred]

        grad_ori = torch.autograd.grad(
            ori_logit,
            adv_image,
            retain_graph=True,
            create_graph=False
        )[0]

        min_distance = float("inf")
        best_perturb = None

        for k in range(num_classes):
            if k == ori_pred:
                continue

            class_logit = logits[k]

            grad_k = torch.autograd.grad(
                class_logit,
                adv_image,
                retain_graph=True,
                create_graph=False
            )[0]

            # 分类边界近似：
            # f_k(x) - f_y(x) = 0
            w_k = grad_k - grad_ori
            f_k = logits[k] - ori_logit

            w_norm = w_k.view(-1).norm(p=2) + 1e-8

            distance = torch.abs(f_k) / w_norm

            if distance.item() < min_distance:
                min_distance = distance.item()

                # r = |f| / ||w||^2 * w
                best_perturb = torch.abs(f_k) / (w_norm ** 2) * w_k

        if best_perturb is None:
            adv_image = adv_image.detach()
            break

        adv_image = adv_image.detach() + (1.0 + overshoot) * best_perturb.detach()

        # 限制最大扰动范围
        delta = torch.clamp(
            adv_image - ori_image,
            min=-eps,
            max=eps
        )

        adv_image = clamp(ori_image + delta, 0.0, 1.0).detach()

    return adv_image.detach()


def deepfool_attack(
    model,
    images,
    labels=None,
    eps=8 / 255,
    max_steps=20,
    overshoot=0.02,
    num_classes=10
):
    """
    批量 DeepFool 攻击。

    注意：
    DeepFool 本质上是逐样本计算边界方向，
    所以这里对 batch 中每张图片逐个攻击。
    """

    adv_list = []

    for i in range(images.size(0)):
        image_i = images[i:i + 1]

        adv_i = deepfool_single(
            model=model,
            image=image_i,
            eps=eps,
            max_steps=max_steps,
            overshoot=overshoot,
            num_classes=num_classes
        )

        adv_list.append(adv_i)

    adv_images = torch.cat(adv_list, dim=0)

    return adv_images.detach()


# =========================================================
# 5. SPSA Attack
# =========================================================
def spsa_attack(
    model,
    images,
    labels,
    eps=8 / 255,
    alpha=2 / 255,
    steps=10,
    samples=16,
    sigma=0.001
):
    """
    SPSA: Simultaneous Perturbation Stochastic Approximation

    黑盒攻击方法：
    不使用反向传播梯度，而是通过查询模型输出估计梯度。

    每一步：
    1. 随机生成方向 u；
    2. 查询 x + sigma*u 和 x - sigma*u；
    3. 根据两次 loss 差异估计梯度；
    4. 更新对抗样本。
    """

    model.eval()

    ori_images = images.clone().detach()
    labels = labels.clone().detach()

    adv_images = ori_images.clone().detach()

    for _ in range(steps):
        grad_estimate = torch.zeros_like(adv_images)

        for _ in range(samples):
            # Rademacher 随机扰动方向：取值为 -1 或 +1
            noise = torch.empty_like(adv_images).bernoulli_(0.5)
            noise = noise * 2 - 1

            images_plus = clamp(adv_images + sigma * noise, 0.0, 1.0)
            images_minus = clamp(adv_images - sigma * noise, 0.0, 1.0)

            with torch.no_grad():
                logits_plus = model(images_plus)
                logits_minus = model(images_minus)

                loss_plus = F.cross_entropy(
                    logits_plus,
                    labels,
                    reduction="none"
                )

                loss_minus = F.cross_entropy(
                    logits_minus,
                    labels,
                    reduction="none"
                )

            # 每个样本单独估计梯度
            coeff = ((loss_plus - loss_minus) / (2.0 * sigma))
            coeff = coeff.view(-1, 1, 1, 1)

            grad_estimate += coeff * noise

        grad_estimate = grad_estimate / float(samples)

        adv_images = adv_images + alpha * grad_estimate.sign()

        delta = torch.clamp(
            adv_images - ori_images,
            min=-eps,
            max=eps
        )

        adv_images = clamp(ori_images + delta, 0.0, 1.0).detach()

    return adv_images.detach()


# =========================================================
# 6. 统一接口 make_adv
# =========================================================
def make_adv(
    attack,
    model,
    images,
    labels,
    eps=8 / 255,
    alpha=2 / 255,
    steps=10,
    max_deepfool_steps=20,
    spsa_samples=16,
    spsa_sigma=0.001,
    random_start=True
):
    """
    统一攻击调用接口。

    attack 可选：
    - fgsm
    - pgd
    - mi-fgsm
    - mifgsm
    - deepfool
    - spsa
    """

    attack = attack.lower()

    if attack == "fgsm":
        adv_images = fgsm_attack(
            model=model,
            images=images,
            labels=labels,
            eps=eps
        )

    elif attack == "pgd":
        adv_images = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            eps=eps,
            alpha=alpha,
            steps=steps,
            random_start=random_start
        )

    elif attack in ["mi-fgsm", "mifgsm", "mi_fgsm"]:
        adv_images = mi_fgsm_attack(
            model=model,
            images=images,
            labels=labels,
            eps=eps,
            alpha=alpha,
            steps=steps,
            decay=1.0
        )

    elif attack == "deepfool":
        adv_images = deepfool_attack(
            model=model,
            images=images,
            labels=labels,
            eps=eps,
            max_steps=max_deepfool_steps,
            overshoot=0.02,
            num_classes=10
        )

    elif attack == "spsa":
        adv_images = spsa_attack(
            model=model,
            images=images,
            labels=labels,
            eps=eps,
            alpha=alpha,
            steps=steps,
            samples=spsa_samples,
            sigma=spsa_sigma
        )

    else:
        raise ValueError(f"未知攻击方法: {attack}")

    return adv_images.detach()