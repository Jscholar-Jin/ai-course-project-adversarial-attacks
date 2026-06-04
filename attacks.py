from dataclasses import dataclass

import torch
import torch.nn.functional as F

from defenses import logits_with_preprocess


def clamp_unit(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


@dataclass
class AttackConfig:
    eps: float = 8 / 255
    alpha: float = 2 / 255
    steps: int = 10
    overshoot: float = 0.02
    max_deepfool_steps: int = 20
    defense: str = "none"
    bpda: bool = False
    jpeg_quality: int = 75
    bit_depth: int = 5

    def defense_kwargs(self):
        return {
            "jpeg_quality": self.jpeg_quality,
            "bit_depth": self.bit_depth,
        }


def model_logits(model, x, config: AttackConfig):
    return logits_with_preprocess(
        model,
        x,
        defense=config.defense,
        bpda=config.bpda,
        **config.defense_kwargs(),
    )


def fgsm(model, images, labels, config: AttackConfig):
    was_training = model.training
    model.eval()

    adv = images.detach().clone()
    adv.requires_grad_(True)

    loss = F.cross_entropy(model_logits(model, adv, config), labels)
    model.zero_grad(set_to_none=True)
    loss.backward()

    adv = clamp_unit(adv + config.eps * adv.grad.sign()).detach()
    model.zero_grad(set_to_none=True)

    if was_training:
        model.train()
    return adv


def pgd(model, images, labels, config: AttackConfig):
    was_training = model.training
    model.eval()

    base = images.detach()
    adv = clamp_unit(base + torch.empty_like(base).uniform_(-config.eps, config.eps))

    for _ in range(config.steps):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model_logits(model, adv, config), labels)
        model.zero_grad(set_to_none=True)
        loss.backward()

        adv = adv + config.alpha * adv.grad.sign()
        delta = torch.clamp(adv - base, min=-config.eps, max=config.eps)
        adv = clamp_unit(base + delta).detach()

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return adv


def deepfool_single(model, image, label, config: AttackConfig):
    base = image.detach()
    adv = base.clone().unsqueeze(0)
    label = label.view(1)

    with torch.no_grad():
        original = model_logits(model, adv, config).argmax(1)
    if original.item() != label.item():
        return adv.squeeze(0)

    for _ in range(config.max_deepfool_steps):
        adv = adv.detach().requires_grad_(True)
        logits = model_logits(model, adv, config)
        pred = logits.argmax(1)
        if pred.item() != label.item():
            break

        current_logit = logits[0, pred.item()]
        model.zero_grad(set_to_none=True)
        current_logit.backward(retain_graph=True)
        grad_orig = adv.grad.detach().clone()

        min_step = None
        best_direction = None

        for cls in range(logits.size(1)):
            if cls == pred.item():
                continue

            adv.grad.zero_()
            model.zero_grad(set_to_none=True)
            logits[0, cls].backward(retain_graph=True)
            grad_cls = adv.grad.detach().clone()

            w = grad_cls - grad_orig
            f = (logits[0, cls] - current_logit).detach()
            norm_w = w.flatten(1).norm(p=2, dim=1).clamp_min(1e-12)
            step = torch.abs(f) / norm_w

            if min_step is None or step.item() < min_step.item():
                min_step = step
                best_direction = w / norm_w.view(-1, 1, 1, 1)

        if best_direction is None:
            break

        perturb = (min_step + 1e-4).view(-1, 1, 1, 1) * best_direction
        adv = adv + (1 + config.overshoot) * perturb
        delta = torch.clamp(adv - base.unsqueeze(0), min=-config.eps, max=config.eps)
        adv = clamp_unit(base.unsqueeze(0) + delta)

    return adv.detach().squeeze(0)


def deepfool(model, images, labels, config: AttackConfig):
    was_training = model.training
    model.eval()
    out = [deepfool_single(model, image, label, config) for image, label in zip(images, labels)]
    if was_training:
        model.train()
    return torch.stack(out, dim=0)


def make_adv(
    name: str,
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 10,
    overshoot: float = 0.02,
    max_deepfool_steps=None,
    defense: str = "none",
    bpda: bool = False,
    jpeg_quality: int = 75,
    bit_depth: int = 5,
):
    key = name.lower().replace("-", "")
    deepfool_steps = steps if max_deepfool_steps is None else max_deepfool_steps
    config = AttackConfig(
        eps=eps,
        alpha=alpha,
        steps=steps,
        overshoot=overshoot,
        max_deepfool_steps=deepfool_steps,
        defense=defense,
        bpda=bpda,
        jpeg_quality=jpeg_quality,
        bit_depth=bit_depth,
    )

    if key == "fgsm":
        return fgsm(model, images, labels, config)
    if key == "pgd":
        return pgd(model, images, labels, config)
    if key == "deepfool":
        return deepfool(model, images, labels, config)
    raise ValueError(f"unknown attack: {name}")
