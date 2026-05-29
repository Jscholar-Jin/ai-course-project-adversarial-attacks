import torch
import torch.nn.functional as F


def clamp(x):
    return torch.clamp(x, 0.0, 1.0)


def fgsm(model, images, labels, eps=8 / 255):
    was_training = model.training
    model.eval()

    adv = images.detach().clone()
    adv.requires_grad = True

    loss = F.cross_entropy(model(adv), labels)
    model.zero_grad(set_to_none=True)
    loss.backward()

    adv = clamp(adv + eps * adv.grad.sign()).detach()
    model.zero_grad(set_to_none=True)

    if was_training:
        model.train()
    return adv


def pgd(model, images, labels, eps=8 / 255, alpha=2 / 255, steps=10):
    was_training = model.training
    model.eval()

    base = images.detach()
    adv = clamp(base + torch.empty_like(base).uniform_(-eps, eps))

    for _ in range(steps):
        adv.requires_grad = True
        loss = F.cross_entropy(model(adv), labels)
        model.zero_grad(set_to_none=True)
        loss.backward()

        adv = adv + alpha * adv.grad.sign()
        delta = torch.clamp(adv - base, min=-eps, max=eps)
        adv = clamp(base + delta).detach()

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return adv


def mifgsm(model, images, labels, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0):
    was_training = model.training
    model.eval()

    base = images.detach()
    adv = base.clone()
    momentum = torch.zeros_like(adv)

    for _ in range(steps):
        adv.requires_grad = True
        loss = F.cross_entropy(model(adv), labels)
        model.zero_grad(set_to_none=True)
        loss.backward()

        grad = adv.grad.detach()
        grad = grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        momentum = decay * momentum + grad

        adv = adv + alpha * momentum.sign()
        delta = torch.clamp(adv - base, min=-eps, max=eps)
        adv = clamp(base + delta).detach()

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return adv


def make_adv(name, model, images, labels, eps=8 / 255, alpha=2 / 255, steps=10):
    key = name.lower().replace("-", "")
    if key == "fgsm":
        return fgsm(model, images, labels, eps=eps)
    if key == "pgd":
        return pgd(model, images, labels, eps=eps, alpha=alpha, steps=steps)
    if key == "mifgsm":
        return mifgsm(model, images, labels, eps=eps, alpha=alpha, steps=steps)
    raise ValueError(f"unknown attack: {name}")
