# Simple ML Project with GitHub CI/CD → Hugging Face

> [!INFO]
> Your delivery should look like this:
> - https://github.com/EmporioSabo/california-housing-predictor
> - https://huggingface.co/EmporioSabo/california-housing-predictor/tree/main
>
> After finishing the task, send the URLs in the WeChat Group.

## Objective

1. Conceive a simple Machine Learning project (no need for complex models).
2. Upload your code to GitHub.
3. Use GitHub CI/CD to automatically train your model and upload it to Hugging Face Hub.


## Inspiration

* [Space Mining GitHub Repo](https://github.com/reveurmichael/space_mining/tree/main)
* [Space Mining Hugging Face Hub](https://huggingface.co/LUNDECHEN/space-mining-ppo/tree/main)
* [GitHub CI/CD Workflow Example](https://github.com/reveurmichael/space_mining/blob/main/.github/workflows/train-long-wandb-hf.yml)

> For simplicity, your project can use a small ML model (like a simple scikit-learn regression or classification).


## Example GitHub CI/CD Workflow

You can use any YAML workflow template you like, as long as your GitHub repository is connected to Hugging Face and the CI/CD pipeline successfully uploads your model. The example might be a simple starting point that might work.

Save this as `.github/workflows/train-and-upload.yml`:

```yaml
name: Train and Upload to Hugging Face

on:
  push:
    branches: [main, master]
  workflow_dispatch:

jobs:
  train-and-upload:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Train model
        run: python train.py

      - name: Upload to Hugging Face Hub
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python - <<'EOF'
          import os
          from huggingface_hub import HfApi

          token = os.environ["HF_TOKEN"]
          repo_id = "YOUR_HF_REPO"
          api = HfApi()

          # Create repo if it doesn't exist
          api.create_repo(repo_id, token=token, exist_ok=True)

          # Upload all model artifacts
          for filename in [FILES_TO_UPLOAD_TO_HF]:
              api.upload_file(
                  path_or_fileobj=filename,
                  path_in_repo=filename,
                  repo_id=repo_id,
                  token=token,
              )
              print(f"Uploaded {filename}")

          print(f"Model uploaded to https://huggingface.co/{repo_id}")
          EOF
```


### Notes

* **HF_TOKEN**: Store your Hugging Face token as a secret in your GitHub repository.
* **YOUR_HF_REPO**: Replace with your Hugging Face repo name.
* **FILES_TO_UPLOAD_TO_HF**: List the model files you want to upload (e.g., `["model.pt", "config.json"]`).
* Workflow triggers:

  * `push` to `main` or  `master`
  * Manual `workflow_dispatch` (can trigger from GitHub UI)
