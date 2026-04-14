# Colab Notebook Cells

Use these cells in order in a fresh Google Colab GPU notebook.

## Cell 1: Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 2: Go To Project Folder

```python
%cd /content/drive/MyDrive/BP Hackathon
!pwd
!ls
```

## Cell 3: Install Packages

```python
!python -m pip install -q -r winning_solution/requirements_colab.txt
```

## Cell 4: Check GPU

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

## Cell 5: Analyze Dataset

```python
!python winning_solution/01_analyze_dataset.py --config winning_solution/configs/safe_deeplabv3plus.yaml
```

## Cell 6: First Training Run

```python
!python winning_solution/02_train.py --config winning_solution/configs/safe_deeplabv3plus.yaml
```

## Cell 7: Evaluate On Validation

```python
!python winning_solution/03_evaluate.py --config winning_solution/configs/safe_deeplabv3plus.yaml --split val
```

## Cell 8: Evaluate On Local Test Split

```python
!python winning_solution/03_evaluate.py --config winning_solution/configs/safe_deeplabv3plus.yaml --split test
```

## Cell 9: Generate Final Prediction Masks

```python
!python winning_solution/04_predict_test.py --config winning_solution/configs/safe_deeplabv3plus.yaml
```

## Cell 10: Open Key Outputs

```python
import pandas as pd

val_metrics = "/content/drive/MyDrive/BP Hackathon/winning_solution/outputs/safe_deeplabv3plus/eval_val/per_class_iou.csv"
test_metrics = "/content/drive/MyDrive/BP Hackathon/winning_solution/outputs/safe_deeplabv3plus/eval_test/per_class_iou.csv"

print("Validation per-class IoU")
display(pd.read_csv(val_metrics))

print("Local test per-class IoU")
display(pd.read_csv(test_metrics))
```

## Cell 11: Show A Few Result Images

```python
from IPython.display import Image, display
from pathlib import Path

comparison_dir = Path("/content/drive/MyDrive/BP Hackathon/winning_solution/outputs/safe_deeplabv3plus/eval_val/comparisons")
for image_path in sorted(comparison_dir.glob("*.png"))[:5]:
    print(image_path.name)
    display(Image(filename=str(image_path)))
```

## Cell 12: Stronger Run If Pretrained Is Allowed

```python
!python winning_solution/02_train.py --config winning_solution/configs/strong_if_pretrained_allowed.yaml
!python winning_solution/03_evaluate.py --config winning_solution/configs/strong_if_pretrained_allowed.yaml --split val
!python winning_solution/03_evaluate.py --config winning_solution/configs/strong_if_pretrained_allowed.yaml --split test
!python winning_solution/04_predict_test.py --config winning_solution/configs/strong_if_pretrained_allowed.yaml
```

## What To Tune First

Tune only one thing at a time.

1. Start with the safe config and record:
   - best validation mIoU
   - local test mIoU
   - per-class IoU

2. If GPU memory is enough, increase train resolution in `safe_deeplabv3plus.yaml`:
   - `image_height: 512`
   - `image_width: 896`
   - if it crashes, reduce `batch_size` to `2`

3. If validation is still improving at the end, increase:
   - `epochs: 55`
   - `patience: 14`

4. If organizers allow pretrained encoders, use:
   - `strong_if_pretrained_allowed.yaml`

5. If the model still confuses classes like sky and flowers:
   - inspect `confusion_matrix.csv`
   - inspect `per_class_iou.csv`
   - look at the saved comparison images

## Best Practical Tuning Order

1. Safe config baseline
2. Higher resolution safe config
3. Longer training safe config
4. Strong pretrained config
5. If memory allows, change strong config inference scales from `[1.0, 1.15]` to `[1.0, 1.1, 1.2]`

## If Colab Runs Out Of Memory

Reduce these in order:

1. `train.batch_size`
2. `train.image_height`
3. `train.image_width`
4. switch from strong config back to safe config

## Most Important Output Files

- `winning_solution/outputs/.../best.ckpt`
- `winning_solution/outputs/.../history.csv`
- `winning_solution/outputs/.../training_curves.png`
- `winning_solution/outputs/.../eval_val/per_class_iou.csv`
- `winning_solution/outputs/.../eval_val/confusion_matrix.csv`
- `winning_solution/outputs/.../eval_test/per_class_iou.csv`
- `winning_solution/outputs/.../submission_predictions/raw_ids/`

