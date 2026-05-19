# transformer_kan

Time-series forecasting models combining Transformer and KAN (Kolmogorov–Arnold Network) architectures, with baseline models (LSTM, GRU, BiLSTM, TCN, etc.).

## Project structure

| File | Description |
|------|-------------|
| `train.py` | Main training script |
| `data_pre.py` | Data preprocessing |
| `model_Trans_KAN.py` | Transformer + KAN model |
| `model_Trans_KAN_large.py` | Larger variant |
| `model_*.py` | Baseline models (LSTM, GRU, Transformer, …) |
| `tool_for_*.py` | Training, testing, and preprocessing utilities |

## Requirements

- Python 3.x
- PyTorch
- NumPy, Pandas, Matplotlib, scikit-learn

Install dependencies according to your environment (e.g. `pip install torch numpy pandas matplotlib scikit-learn`).

## Usage

```bash
python train.py
```

Training logs are written under `output/` (ignored by git).

## Upload to GitHub

This machine cannot use `git push` directly (Git is not installed; `github.com` may be unreachable). Use the included uploader (calls `api.github.com` only):

1. Create a token: [GitHub → Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens) with **repo** scope.
2. In PowerShell, from this folder:

```powershell
$env:GITHUB_TOKEN = "ghp_你的token"
d:\cursor\cursor\resources\app\resources\helpers\node.exe upload_to_github.mjs
```

Or double-click `upload_github.bat` after setting `GITHUB_TOKEN` in the environment.

This creates **`transformer_kan`** on your account and uploads source files (training logs under `output/` are ignored).

## License

Add a license if you plan to open-source this repository.
