# Colab Run Guide

This pipeline is a stronger replacement for the provided sample scripts.

It fixes the biggest starter-code issues:

- uses the correct 10 classes from the PDF
- keeps `Flowers (600)` instead of dropping it
- avoids corrupting masks during resize
- trains the full segmentation model instead of only a tiny head
- saves proper validation and test predictions in raw label IDs

## Important metric note

`0.80` mean IoU is not the same thing as `80%` pixel accuracy.

- IoU is class overlap: `TP / (TP + FP + FN)`
- mean IoU averages classes more fairly than pixel accuracy
- sky and landscape can look good while flowers or logs are still failing badly

## What I found in your real data

- Starter baseline shown by Duality: `0.2478` mean IoU.
- The provided training code has a label bug: it removes `Flowers (600)`.
- `Logs` are extremely rare in train and val.
- The local test split has only a subset of classes present, so local test IoU and hidden-test IoU may differ.

## Folder expectation

Point `dataset.root_dir` in the config to the folder that contains:

```text
train/
val/
Offroad_Segmentation_testImages/
```

Each split should contain:

```text
Color_Images/
Segmentation/
```

## Colab steps

1. Upload the whole project folder to Google Drive, or mount Drive where this folder lives.
2. Open a Colab notebook and set the runtime to GPU.
3. `cd` into the project root.
4. Install requirements.
5. Run dataset analysis.
6. Train with the safe config first.
7. Evaluate on val.
8. Evaluate on the local test split.
9. Generate submission-style predictions.

## Exact Colab commands

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive/BP Hackathon
!python -m pip install -r winning_solution/requirements_colab.txt
```

```python
!python winning_solution/01_analyze_dataset.py --config winning_solution/configs/safe_deeplabv3plus.yaml
```

```python
!python winning_solution/02_train.py --config winning_solution/configs/safe_deeplabv3plus.yaml
```

```python
!python winning_solution/03_evaluate.py --config winning_solution/configs/safe_deeplabv3plus.yaml --split val
```

```python
!python winning_solution/03_evaluate.py --config winning_solution/configs/safe_deeplabv3plus.yaml --split test
```

```python
!python winning_solution/04_predict_test.py --config winning_solution/configs/safe_deeplabv3plus.yaml
```

## If organizers allow pretrained encoders

Use the stronger config:

```python
!python winning_solution/02_train.py --config winning_solution/configs/strong_if_pretrained_allowed.yaml
```

Then evaluate it the same way:

```python
!python winning_solution/03_evaluate.py --config winning_solution/configs/strong_if_pretrained_allowed.yaml --split val
!python winning_solution/03_evaluate.py --config winning_solution/configs/strong_if_pretrained_allowed.yaml --split test
```

## Outputs you should watch

- `winning_solution/outputs/.../history.csv`
- `winning_solution/outputs/.../training_curves.png`
- `winning_solution/outputs/.../best.ckpt`
- `winning_solution/outputs/.../eval_val/metrics.txt`
- `winning_solution/outputs/.../eval_test/metrics.txt`
- `winning_solution/outputs/.../submission_predictions/raw_ids/`

## Best practical training order

1. Run the safe config once and record val/test mean IoU.
2. If memory allows, increase `train.image_height` and `train.image_width`.
3. Increase epochs if validation keeps improving.
4. Turn on the stronger pretrained config only if it is allowed by organizers.
5. Compare both on `val` and on the local `test` split.
6. Keep the checkpoint with the best unseen performance, not just the best-looking train curve.

## What to put in the report

- initial baseline: `0.2478` mean IoU from Duality
- bug fixes in starter code
- your model choice and why
- augmentation and loss strategy
- class imbalance analysis
- before/after prediction examples
- per-class IoU on val and local test

## Very important

Do not train on `Offroad_Segmentation_testImages`.

Use it only for evaluation or prediction.

