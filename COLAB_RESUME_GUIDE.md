# Winning Solution V3: Colab Resume Guide

This folder is a separate stronger experiment. Your earlier runs stay untouched.

## Goal

- stronger augmentation and rare-class-balanced training
- separate output folder
- automatic resume from `last.ckpt` if Colab disconnects

## Important behavior

`02_train.py` now supports resume.

- If `last.ckpt` exists in the configured `save_dir`, training resumes automatically.
- It resumes from the **last completed epoch**, not the exact middle of a crashed epoch.
- `best.ckpt` is still the best validation model.

## First run in Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive/BP Hackathon/winning_solution_v3
!python -m pip install -q -r requirements_colab.txt
```

```python
!python 01_analyze_dataset.py --config configs/exp3_final_last_run.yaml
```

```python
!python 02_train.py --config configs/exp3_final_last_run.yaml --resume auto
```

## If Colab disconnects or GPU quota ends

Reconnect and run only these again:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive/BP Hackathon/winning_solution_v3
!python -m pip install -q -r requirements_colab.txt
```

```python
!ls outputs/exp3_final_last_run
```

```python
!python 02_train.py --config configs/exp3_final_last_run.yaml --resume auto
```

If `last.ckpt` is present, training will continue from the next epoch automatically.

## After training

```python
!python 03_evaluate.py --config configs/exp3_final_last_run.yaml --split val
!python 03_evaluate.py --config configs/exp3_final_last_run.yaml --split test
!python 04_predict_test.py --config configs/exp3_final_last_run.yaml
```

## What to compare against the old run

Compare these with `winning_solution/outputs/safe_deeplabv3plus` and `winning_solution_v2/outputs/exp2_resnet101_pretrained_resume`:

- validation mean IoU
- local test mean IoU
- `Rocks` IoU
- `Dry Bushes` IoU
- whether false predictions for absent classes go down
- whether `Flowers`, `Logs`, and `Ground Clutter` false positives go down on local test

Extra outputs:

- `eval_val/per_image_scores.csv`
- `eval_test/per_image_scores.csv`
- all comparison images are saved because `save_comparisons: -1`
