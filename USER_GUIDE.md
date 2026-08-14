# 📘 ForgeML User Guide

This guide walks you through setting up and using ForgeML v2.0.

ForgeML v2.0 is a generalized, production-ready MLOps platform. You can run any arbitrary ML training script on Kaggle by simply specifying it as the `entrypoint` in your configuration.

---

## Step 1: Source Code Directory Structure

Your project directory must follow the IndustrialAD structure:

```text
MyNewProject/
├── src/                <-- Contains all Python code (e.g., models, dataset loaders, utils)
├── configs/            <-- (Optional) Contains hyperparameter YAML/JSON configuration files
└── requirements.txt    <-- Lists dependencies to install (e.g., torch, numpy, scikit-learn)
```
*Note: ForgeML will automatically install the libraries listed in `requirements.txt` on the Kaggle virtual machine before execution.*

---

## Step 2: Initialize the Project (`forge init`)

Open your terminal, navigate to your project directory (with the source code from Step 1), and run the initialization command:

```bash
cd /path/to/MyNewProject
forge init
```

This command will generate a configuration file named `forge.yaml` in your project directory.

---

## Step 3: Configure Kaggle Resources

Open the newly created `forge.yaml` file. Its content will look like this:

```yaml
project:
  name: my_project_name              # Your project name

provider:
  name: kaggle

kaggle:
  kernel: my-training-kernel         # The Kaggle Kernel name (customizable)
  dataset: my-source-dataset         # The Private Dataset name for your code (customizable)
  mvtec_dataset: "ipythonx/mvtec-ad" # <-- Public Kaggle dataset for training data
  accelerator: NvidiaTeslaT4
  internet: true
  entrypoint: scripts/train.py       # <-- The script in your source code to execute on Kaggle
```

**Important configurations:** 
- **`entrypoint`**: This defines the exact Python script inside your `src` folder that Kaggle should run. It gives you complete freedom to run any ML workload.
- **`mvtec_dataset`** (or your custom dataset key): Paste the **dataset slug** from Kaggle here (e.g., `johndoe/cats-and-dogs`). ForgeML will automatically mount this massive dataset to your Kernel without requiring you to download it locally!

---

## Step 4: Trigger Training (`forge run`)

When your code is ready, use a single command to send everything to the cloud:

```bash
forge run --model <model-name> --category <data-label>
```
*(These variables will be passed directly into the `run_config.json` file for your source code in `src` to read and process).*

This process is fully automated:
1. Packages the source code into `bundle.tar.gz`.
2. Creates a Private Dataset on Kaggle and uploads the source code.
3. Initializes the Script Kernel and starts training.
4. Automatically collects weight files (`.pkl`/`.pth`) and `metrics.json` from Kaggle, downloading them straight to the `artifacts/<run_id>/output/` directory on your machine!

---

## Step 5: Monitoring & Management

While the Kernel is running on Kaggle (which can take anywhere from tens of minutes to a few hours), you can check the status using the following commands:

- **Check current status:**
  ```bash
  forge status
  ```
  *(Displays statuses like `PACKAGING`, `UPLOADING`, `QUEUED`, `RUNNING`, `COMPLETED`...)*

- **View run history (Tracking):**
  ```bash
  forge history
  ```
  *(Displays a summary table containing Run IDs, Models, Run Times, and Metrics).*

---

## 🤖 Advanced Features

### 1. Resume Process
If you accidentally close your terminal while waiting for Kaggle to run, don't worry! ForgeML continues to monitor in the background. You just need to retrieve the `Run ID` (using `forge history`) and resume:
```bash
forge run --run-id "20260809T103602Z-687f"
```
ForgeML will skip the code upload step and download the results once Kaggle finishes.

### 2. Natural Language Interface
If you have configured `OPENAI_API_KEY` in your project's `.env` file, you can command ForgeML using natural language:
```bash
forge ask "Train a fastflow model on the pill dataset with seed 123"
```
The AI will automatically parse your prompt into accurate configuration parameters and ask for confirmation before execution!

## 📦 Packaging & Release (Release v2.0)

ForgeML supports building as a standalone Python module. To create installation files (`.whl` and `.tar.gz`):

```bash
cd /path/to/ForgeML
pip install build
python -m build
```
The installation files will be located in the `dist/` directory. You can share these files, and anyone can install it by running `pip install forgeml-0.1.0-py3-none-any.whl` to use the `forge` command anywhere!
