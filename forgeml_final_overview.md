# ForgeML

## Final Project Overview

**Status:** Planning
**Primary execution provider:** Kaggle
**Local environment:** Ubuntu machine without a GPU
**Initial workload:** IndustrialAD / MVTec Anomaly Detection
**LLM planner:** One fixed provider through its official Python SDK
**Pi status:** Deferred to Future Work unless polyglot/IPC experience is an explicit portfolio goal

## 1. Product Definition

ForgeML automates the repetitive loop between local ML development and remote
training on Kaggle.

```text
Edit local source
      ↓
Package a reproducible run
      ↓
Upload to Kaggle
      ↓
Run a fixed Kernel entrypoint
      ↓
Monitor execution
      ↓
Download artifacts
      ↓
Record the experiment
```

The first user-facing workflow is:

```bash
forge run --model patchcore --dataset mvtec --category bottle
```

Natural language is an optional interface over the same workflow:

```bash
forge ask "Train PatchCore on Bottle"
```

ForgeML is not an ML framework, a replacement for Kaggle, or a general MLOps
platform. It is a focused control layer for one real workload and one provider.

## 2. Core Design Rules

1. Prove the Kaggle pipeline before building the LLM, database, or UI.
2. Keep the Kernel entrypoint fixed; pass experiment-specific values through a
   versioned configuration contract.
3. Keep source code, experiment parameters, and generated artifacts separate.
4. Let the LLM create a plan only. ForgeML validates and executes that plan.
5. Start with one provider, one workflow, one notification channel, and one
   workload.
6. Record enough lineage to reproduce every run.

## 3. Architecture

```text
                         User
                          │
             ┌────────────┴────────────┐
             │                         │
       forge run                 forge ask
             │                         │
             │              Fixed-provider SDK
             │              (Python, structured output)
             │                         │
             └──────────────┬──────────┘
                            ▼
                    ExecutionPlan JSON
                            │
                   Pydantic validation
                            │
                    Capability validation
                            │
                     Workflow state machine
                            │
                    Project/package manager
                            │
                      Kaggle Provider
                 ┌──────────┴──────────┐
                 ▼                     ▼
          Dataset Manager        Kernel Manager
                 │                     │
                 └──────────┬──────────┘
                            ▼
                    Fixed Kernel entrypoint
                            │
                       Training run
                 ┌──────────┴──────────┐
                 ▼                     ▼
             Artifacts             Metrics/logs
                 │                     │
                 └──────────┬──────────┘
                            ▼
                    SQLite + notification
```

ForgeML Python remains the execution authority. The planner cannot call
Kaggle, execute shell commands, mutate source files, or send notifications.

## 4. Kaggle Execution Contract

Kaggle uses two linked resources:

- A private Dataset containing a source bundle, dependency metadata, and one
  `run_config.json`.
- A fixed Kernel entrypoint that loads the Dataset, validates the config, runs
  training, and writes a manifest plus artifacts.

The source should be uploaded as an archive or with explicit directory handling.
Do not assume that nested local directories are preserved by a default Dataset
upload.

Example input contract:

```json
{
  "schema_version": 1,
  "run_id": "20260808T164500Z-7f3c",
  "action": "train",
  "project": "industrialad",
  "source": {
    "git_commit": "abc123",
    "bundle_sha256": "..."
  },
  "training": {
    "model": "patchcore",
    "dataset": "mvtec",
    "category": "bottle",
    "config_path": "configs/patchcore.yaml",
    "seed": 42
  },
  "output": {
    "metrics_file": "metrics.json",
    "artifact_dir": "artifacts"
  }
}
```

The Kernel must fail before training when the schema, model, category, source
hash, or dependency contract is invalid. It must write `run_manifest.json` with
the resolved config, package versions, runtime metadata, and output file hashes.

Only one active run is allowed per project in v1. This avoids races between a
new Dataset version, the latest Kernel status, and output retrieval. Every local
run gets an immutable `run_id` and an exclusive project lock.

## 5. Workflow State Machine

The first workflow engine should be a small explicit state machine, not a generic
orchestration framework:

```text
CREATED
  → PACKAGING
  → DATASET_UPLOADING
  → DATASET_READY
  → KERNEL_SUBMITTING
  → QUEUED
  → RUNNING
  → COLLECTING
  → COMPLETED
```

Failure states are explicit:

```text
FAILED_CONFIG
FAILED_DEPENDENCY
FAILED_EXECUTION
FAILED_ARTIFACT
BLOCKED_QUOTA
BLOCKED_AUTH
FAILED_TRANSIENT
```

Only network/API failures and selected transient runtime failures may be retried.
Configuration errors, import errors, CUDA OOM, and quota exhaustion must not be
blindly retried.

## 6. Natural-Language Planning

Phase 4 uses one fixed LLM provider through its official Python SDK. Structured
output/tool use produces an `ExecutionPlan`; Pydantic validates it again inside
ForgeML. This keeps the MVP in one language and avoids a second runtime, sidecar
lifecycle, custom IPC protocol, and duplicate dependency tree.

The planner has no shell, filesystem, Kaggle, or notification tools. It only
returns the four required fields plus a schema version and optional explanation.

```text
Natural-language request
          ↓
Fixed-provider Python SDK
          ↓
ExecutionPlan JSON
          ↓
ForgeML Python client
          ↓
Pydantic + capability validation
          ↓
Same workflow as forge run
```

Planner contract:

```json
{
  "schema_version": 1,
  "action": "train",
  "model": "patchcore",
  "dataset": "mvtec",
  "category": "bottle",
  "config": "configs/patchcore.yaml",
  "reason": "Requested PatchCore training for the Bottle category"
}
```

The planner receives a capability catalog generated by ForgeML. It may select
only valid models, datasets, categories, and config files. It may not choose
shell commands, Kaggle identifiers, accelerator settings, artifact paths,
retry policies, or credentials.

Pi (`@earendil-works/pi-ai`) is intentionally deferred. It is a real and active
library, but its provider normalization, streaming, OAuth, Node runtime, sidecar
lifecycle, and IPC surface are unnecessary for a single fixed provider and one
JSON planning call. Reconsider it only when polyglot/IPC experience is itself an
explicit portfolio objective, or when ForgeML genuinely needs multiple providers.

The following are outside ForgeML v1:

- `pi-coding-agent`;
- `pi-server` and remote Pi sessions;
- Pi extensions or unreviewed packages;
- shell, filesystem, Kaggle, or notification tools exposed to the planner;
- multi-turn agent autonomy.

`pi-agent-core` can be evaluated later if plan clarification requires a stateful
conversation. It is not required for the initial product.

## 7. Phase 0: Kaggle Pipeline Proof

Do not build ForgeML first. Prove this pipeline manually:

```text
IndustrialAD source
      ↓
Source bundle + run_config.json
      ↓
Private Kaggle Dataset
      ↓
Fixed Kernel entrypoint
      ↓
Import source and install locked dependencies
      ↓
Run PatchCore + Bottle
      ↓
Write metrics, checkpoint, and run_manifest.json
      ↓
Download output locally
```

Phase 0 is complete only when:

- source directories arrive intact on Kaggle;
- the Kernel reads and validates `run_config.json`;
- dependencies install deterministically;
- Internet and accelerator settings are explicit;
- at least two models run through the same Kernel;
- a source update changes the bundle hash and executed code;
- metrics and artifacts download into the correct `run_id` directory;
- a failed config is rejected before training.

## 8. Implementation Phases

### Phase 1: Foundation

Build:

- Typer CLI;
- YAML/Pydantic configuration;
- structured logging;
- project inspection and capability catalog;
- explicit workflow state machine;
- local lock and run directory.

Initial commands:

```bash
forge init
forge validate
forge status
forge run --model patchcore --dataset mvtec --category bottle
```

### Phase 2: Kaggle Provider

Implement Dataset Manager, Kernel Manager, status polling, timeout handling,
output collection, and error classification. Persist every provider operation
with a redacted request/response summary.

### Phase 3: Experiment Tracking

Use SQLite and SQLModel for a small run repository:

```text
runs
  id, project, provider, status
  model, dataset, category
  git_commit, bundle_sha256
  kaggle_dataset_version, kaggle_run_id
  started_at, finished_at
  metrics_json, artifact_path, error_type
```

Artifacts are stored under `artifacts/<run_id>/`, never in one shared mutable
directory.

### Phase 4: Natural Language

Add one official Python provider SDK with structured output. `forge ask` must
produce the same validated `ExecutionPlan` used by `forge run`. If the LLM is
disabled or unavailable, explicit CLI execution must continue to work.

### Phase 5: Reliability

Add idempotency, project locks, retry policy, quota detection, dependency
diagnostics, artifact integrity checks, and recovery after process interruption.

### Phase 6: v1 Release

Complete end-to-end tests on IndustrialAD, documentation, installation, demo,
troubleshooting, credential review, and a reproducible release package.

## 9. Project Layout

```text
forgeml/
├── src/forgeml/
│   ├── cli/
│   ├── config/
│   ├── core/                 # state machine, errors, logging
│   ├── project/              # inspection, packaging, hashes
│   ├── workflow/             # plan validation and execution
│   ├── providers/kaggle/     # dataset, kernel, monitor
│   ├── experiments/          # SQLite/SQLModel repository
│   ├── artifacts/
│   ├── notifications/
│   └── ai/                   # planner client and capability catalog
├── templates/kernel_entrypoint.py
├── tests/
├── examples/
├── docs/
├── forge.yaml
├── pyproject.toml
└── .env.example
```

## 10. Configuration and Secrets

```yaml
project:
  name: industrialad

provider:
  name: kaggle

kaggle:
  kernel: industrialad-training
  dataset: industrialad-source
  accelerator: NvidiaTeslaT4

training:
  default_model: patchcore
  default_dataset: mvtec

artifacts:
  directory: artifacts

ai:
  planner: python-sdk
  enabled: false
  provider: openai
  model: fixed-model-name

notifications:
  discord: false
```

Secrets never belong in YAML or source control.

- Kaggle credentials stay in the standard local Kaggle credential store.
- The Python planner process may receive only the selected LLM API key.
- ForgeML owns Kaggle credentials and Discord webhook values.
- `.env`, `.kaggle/`, logs, and artifacts are ignored by Git.
- A leaked credential is revoked and rotated; deleting the file is not enough.

## 11. Testing Strategy

Unit tests cover config, plan schema, capability validation, state transitions,
retry classification, artifact paths, and database operations.

Integration tests cover a small Kaggle job and are separated from normal tests so
they do not consume GPU quota unexpectedly. The planner is tested with a fake
provider response and schema-invalid responses; no paid LLM call is required for
the default test suite.

The most important acceptance test is end-to-end:

```text
forge ask
  → validated plan
  → run_config.json
  → Dataset update
  → fixed Kernel
  → monitored run
  → verified artifacts
  → SQLite record
```

## 12. Definition of Done

ForgeML v1 is complete when a user can run:

```bash
forge ask "Train PatchCore on Bottle"
```

and receive a reproducible result without manually editing or starting Kaggle.
The result must include the validated plan, source commit/hash, Kaggle run
identity, status history, metrics, artifact manifest, and a clear failure reason
when the run does not complete.

The first milestone is not an autonomous AI agent. It is this stable pipeline:

```text
IndustrialAD
    ↓
Kaggle Dataset
    ↓
run_config.json
    ↓
Fixed Kernel Entrypoint
    ↓
Training
    ↓
Artifacts
    ↓
Local experiment record
```

Pi is not required for the first milestone. It can be revisited as Future Work
when its additional runtime and IPC complexity have a concrete product or
portfolio justification.
