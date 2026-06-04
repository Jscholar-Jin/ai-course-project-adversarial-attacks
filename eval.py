import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm

from attacks import make_adv
from data_utils import build_cifar10_loaders
from defenses import logits_with_preprocess
from models import load_model


@dataclass
class EvalTarget:
    name: str
    model: torch.nn.Module
    source_name: str
    defense: str = "none"
    bpda: bool = False
    jpeg_quality: int = 75
    bit_depth: int = 5

    def defense_kwargs(self):
        return {
            "jpeg_quality": self.jpeg_quality,
            "bit_depth": self.bit_depth,
        }


def parse_csv_list(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def predict(model, images, defense="none", bpda=False, **kwargs):
    with torch.no_grad():
        logits = logits_with_preprocess(
            model,
            images,
            defense=defense,
            bpda=bpda,
            **kwargs,
        )
    return logits.argmax(1)


def score_attack(
    attack_name: str,
    source: EvalTarget,
    target: EvalTarget,
    loader,
    args,
    device,
    eps: float,
    steps: int,
    adaptive: bool = False,
):
    source.model.eval()
    target.model.eval()

    total = 0
    clean_ok = 0
    adv_ok = 0
    attack_success = 0
    distances_linf = []
    distances_l2 = []

    iterator = tqdm(loader, desc=f"{source.name}->{target.name} {attack_name}", leave=False)
    for images, labels in iterator:
        if args.max_samples and total >= args.max_samples:
            break

        images = images.to(device)
        labels = labels.to(device)
        if args.max_samples and total + labels.size(0) > args.max_samples:
            keep = args.max_samples - total
            images = images[:keep]
            labels = labels[:keep]

        clean_pred = predict(
            target.model,
            images,
            defense=target.defense,
            bpda=False,
            **target.defense_kwargs(),
        )

        attack_defense = target.defense if adaptive else source.defense
        attack_bpda = adaptive and target.defense != "none"
        attack_kwargs = target.defense_kwargs() if adaptive else source.defense_kwargs()

        adv_images = make_adv(
            attack_name,
            source.model,
            images,
            labels,
            eps=eps,
            alpha=args.alpha,
            steps=steps,
            max_deepfool_steps=args.deepfool_steps,
            defense=attack_defense,
            bpda=attack_bpda,
            jpeg_quality=attack_kwargs["jpeg_quality"],
            bit_depth=attack_kwargs["bit_depth"],
        )

        adv_pred = predict(
            target.model,
            adv_images,
            defense=target.defense,
            bpda=False,
            **target.defense_kwargs(),
        )

        clean_mask = clean_pred.eq(labels)
        adv_mask = adv_pred.eq(labels)
        batch = labels.size(0)
        delta = (adv_images - images).detach()

        total += batch
        clean_ok += clean_mask.sum().item()
        adv_ok += adv_mask.sum().item()
        attack_success += (clean_mask & ~adv_mask).sum().item()
        distances_linf.append(delta.abs().amax(dim=(1, 2, 3)).cpu())
        distances_l2.append(delta.flatten(1).norm(p=2, dim=1).cpu())

    linf = torch.cat(distances_linf)[:total] if distances_linf else torch.tensor([0.0])
    l2 = torch.cat(distances_l2)[:total] if distances_l2 else torch.tensor([0.0])

    return {
        "source": source.name,
        "target": target.name,
        "source_model": source.source_name,
        "target_model": target.source_name,
        "attack": attack_name,
        "adaptive": adaptive,
        "attack_defense": attack_defense,
        "eval_defense": target.defense,
        "bpda": attack_bpda,
        "eps": eps,
        "eps_255": eps * 255,
        "alpha": args.alpha,
        "alpha_255": args.alpha * 255,
        "steps": steps,
        "samples": total,
        "clean_acc": 100.0 * clean_ok / max(total, 1),
        "adv_acc": 100.0 * adv_ok / max(total, 1),
        "asr": 100.0 * attack_success / max(clean_ok, 1),
        "mean_linf": linf.mean().item(),
        "mean_l2": l2.mean().item(),
    }


def load_targets(args, device) -> Dict[str, EvalTarget]:
    required = [args.cnn, args.resnet]
    for path in required:
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing checkpoint: {path}")

    targets = {
        "cnn": EvalTarget(
            name="cnn",
            model=load_model("cnn", args.cnn, device),
            source_name="cnn",
        ),
        "resnet": EvalTarget(
            name="resnet",
            model=load_model("resnet", args.resnet, device),
            source_name="resnet",
        ),
        "cnn_jpeg": EvalTarget(
            name="cnn_jpeg",
            model=load_model("cnn", args.cnn, device),
            source_name="cnn",
            defense="jpeg",
            jpeg_quality=args.jpeg_quality,
        ),
        "cnn_squeeze": EvalTarget(
            name="cnn_squeeze",
            model=load_model("cnn", args.cnn, device),
            source_name="cnn",
            defense="squeeze",
            bit_depth=args.bit_depth,
        ),
    }
    if os.path.exists(args.cnn_adv):
        targets["cnn_adv"] = EvalTarget(
            name="cnn_adv",
            model=load_model("cnn", args.cnn_adv, device),
            source_name="cnn",
        )
    return targets


def build_jobs(targets: Dict[str, EvalTarget], eps_list: List[float], step_list: List[int], args):
    jobs = []

    white_source = targets["cnn"]
    white_target = targets["cnn"]
    transfer_source = targets["resnet"]
    transfer_target = targets["cnn"]

    for attack in args.attacks:
        if attack in {"fgsm", "pgd"}:
            for eps in eps_list:
                jobs.append(("white", attack, white_source, white_target, eps, args.steps, False))
        else:
            jobs.append(("white", attack, white_source, white_target, args.eps, args.steps, False))

        if attack == "pgd":
            for steps in step_list:
                jobs.append(("white_steps", attack, white_source, white_target, args.eps, steps, False))

    for attack in args.attacks:
        jobs.append(("transfer", attack, transfer_source, transfer_target, args.eps, args.steps, False))

    if "cnn_adv" in targets:
        for attack in args.attacks:
            jobs.append(
                ("defense_advtrain", attack, targets["cnn_adv"], targets["cnn_adv"], args.eps, args.steps, False)
            )

    defense_targets = [targets["cnn_jpeg"], targets["cnn_squeeze"]]
    for attack in args.attacks:
        for target in defense_targets:
            jobs.append(("defense_preprocess", attack, white_source, target, args.eps, args.steps, False))

    for target in [targets["cnn_jpeg"], targets["cnn_squeeze"]]:
        jobs.append(("adaptive", "pgd", target, target, args.eps, args.steps, True))

    return jobs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--result-dir", default="./results")
    parser.add_argument("--cnn", default="./checkpoints/cnn.pt")
    parser.add_argument("--resnet", default="./checkpoints/resnet.pt")
    parser.add_argument("--cnn-adv", default="./checkpoints/cnn_adv.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--alpha", type=float, default=2 / 255)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--deepfool-steps", type=int, default=20)
    parser.add_argument("--eps-list", default="2/255,4/255,8/255,12/255")
    parser.add_argument("--step-list", default="1,3,5,10")
    parser.add_argument("--attacks", nargs="+", default=["fgsm", "pgd", "deepfool"])
    parser.add_argument("--bit-depth", type=int, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default="metrics.csv")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def parse_fraction(text: str) -> float:
    text = text.strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    return float(text)


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = build_cifar10_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        args.download,
        test_samples=args.max_samples,
    )
    targets = load_targets(args, device)

    eps_list = parse_csv_list(args.eps_list, parse_fraction)
    step_list = parse_csv_list(args.step_list, int)
    jobs = build_jobs(targets, eps_list, step_list, args)

    rows = []
    for scenario, attack, source, target, eps, steps, adaptive in jobs:
        row = score_attack(
            attack,
            source,
            target,
            test_loader,
            args,
            device,
            eps,
            steps,
            adaptive=adaptive,
        )
        row["scenario"] = scenario
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(args.result_dir, args.output)
    df.to_csv(path, index=False)
    print(df.head(20))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
