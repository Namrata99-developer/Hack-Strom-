## Drive Transfer Guide

You do not need to upload the full dataset again.

### Best option

Keep the dataset folder already present in Google Drive:

- `MyDrive/BP Hackathon/train`
- `MyDrive/BP Hackathon/val`
- `MyDrive/BP Hackathon/Offroad_Segmentation_testImages`

Only upload this code folder:

- `winning_solution_v3`

### If you are switching to another Google account

1. In the old Google Drive account, share the full `BP Hackathon` folder with the new account.
2. In the new account, open **Shared with me**.
3. Add a shortcut of `BP Hackathon` into **MyDrive**.
4. In Colab, use the shortcut path:

```python
%cd /content/drive/MyDrive/BP Hackathon/winning_solution_v3
```

This avoids re-uploading the dataset.

### If uploading the code folder manually

Upload only:

- `01_analyze_dataset.py`
- `02_train.py`
- `03_evaluate.py`
- `04_predict_test.py`
- `requirements_colab.txt`
- `README_COLAB.md`
- `COLAB_RESUME_GUIDE.md`
- `configs/`
- `hackathon_seg/`

Do not upload:

- `outputs/`
- `__pycache__/`

### Resume after disconnect

Training can resume from:

- `outputs/exp3_final_last_run/last.ckpt`

Run:

```python
!python 02_train.py --config configs/exp3_final_last_run.yaml --resume auto
```
