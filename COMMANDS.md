# 💻 ForgeML Command Reference

This document provides a quick reference for all available `forge` CLI commands and their usage.

## Basic Commands

### `forge init`
Initializes a new ForgeML project in the current directory.
- **Description:** Generates a default `forge.yaml` configuration file to define Kaggle settings, dataset links, and project metadata.
- **Usage:** 
  ```bash
  forge init
  ```

### `forge run`
Triggers the model training pipeline on Kaggle.
- **Description:** Packages code, uploads it to Kaggle, starts the training kernel (executing the custom `entrypoint` defined in `forge.yaml`), and waits to download the results.
- **Usage:**
  ```bash
  forge run --model <model-name> --category <data-label>
  ```
- **Resume an interrupted run:**
  ```bash
  forge run --run-id "<run-id>"
  ```
  *(Note: Original configuration flags are automatically restored. You only need to provide flags if you intend to override them).*

## Monitoring & Tracking

### `forge status`
Checks the real-time status of the current or most recent run.
- **Description:** Displays pipeline stages like `PACKAGING`, `UPLOADING`, `QUEUED`, `RUNNING`, or `COMPLETED`.
- **Usage:**
  ```bash
  forge status
  ```

### `forge history`
Displays a comprehensive history of all training runs.
- **Description:** Lists past runs from the local SQLite database (`forge.sqlite`), including Run ID, Model (or Entrypoint for generic workflows), Execution Time, and up to 2 captured Metrics.
- **Usage:**
  ```bash
  forge history
  ```

## Advanced Commands

### `forge ask`
Uses an integrated LLM to translate natural language into training commands.
- **Description:** Parses your request and generates the necessary configuration parameters. Requires an `OPENAI_API_KEY` in the `.env` file. *Note: Currently specifically scoped to the legacy `kaggle_adapter.py` workflows.*
- **Usage:**
  ```bash
  forge ask "Train a fastflow model on the pill dataset with seed 123"
  ```
