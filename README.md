# CIFAR-10 Adversarial Attacks

This project trains two CIFAR-10 classifiers and compares adversarial attacks:

- white-box attack: `cnn -> cnn`
- transfer attack: `resnet -> cnn`
- attacks: `FGSM`, `PGD`, `MI-FGSM`

The main metric is how much accuracy drops on adversarial images. The scripts also
report ASR, the attack success rate on samples that were classified correctly
before the attack.

## Files

```text
models.py    model definitions
attacks.py   FGSM, PGD, MI-FGSM
train.py     train resnet/cnn checkpoints
eval.py      run white-box and transfer attacks
summary.py   merge results and draw figures
```

Generated files go to `checkpoints/`, `logs/`, `results/`, and `figures/`.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Download CIFAR-10 on the first run:

```bash
python train.py --model both --download
```

After checkpoints are saved:

```bash
python eval.py
python summary.py
```

For a quick smoke run:

```bash
python train.py --model both --epochs 1 --download
python eval.py --max-samples 256
```

## Defaults

- data: `./data`
- checkpoints: `./checkpoints/resnet.pt`, `./checkpoints/cnn.pt`
- perturbation: `eps = 8/255`, `alpha = 2/255`, `steps = 10`
- result CSV: `./results/attacks.csv`
