# Kolobok – Automated Tire Valuation Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dockerized](https://img.shields.io/badge/Docker-blue?logo=docker&logoColor=white)](docker-compose.yaml)
[![Telegram Bot](https://img.shields.io/badge/TGBot-blue?logo=telegram&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](#)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?logo=onnx&logoColor=white)](#)
[![React](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black)](#)
[![Tauri](https://img.shields.io/badge/Tauri-24C8DB?logo=tauri&logoColor=white)](#)
[![Tor Network](https://img.shields.io/badge/Tor-7D4698?logo=torproject&logoColor=white)](#)
[![Caddy TLS](https://img.shields.io/badge/Caddy-1F88C0?logo=caddy&logoColor=white)](#)

Kolobok is an end-to-end system that **estimates tread depth, spike wear and recognises tire make / model** from only two user photos

---

## Table of contents
1. [Features](#features)
2. [Architecture](#architecture)
3. [Metrics](#metrics)
4. [Deployment](#deployment)
5. [Reproducibility](#reproducibility)
6. [Configuration](#configuration)
7. [License](#license)

---

## Features
* 📸 **Two-photo workflow** – side wall + tread.
* ⚙️ **Fully dockerised** – `docker-compose up` spins up everything.
* 🤖 **Telegram bot** for a friction-less user interface (Russian UI).
* 🌐 **Frontend** multiple clients:
  * **Web** - React App
  * **Desktop** - Tauri App
  * **Telegram** - Telegram Bot
* ⚡ **CV & ML back-end**
  * Side-wall OCR → brand / model / size extraction
  * Tread-depth regression
  * Spike detector & classifier with visual annotations
* 🔐 **Security**
  * **TLS** backend prontection
  * **Token-based authentication** client and API.
  * **TOR** proxy for outgoing traffic.

## Architecture
Components:
* **`ml`** – FastAPI app exposing the CV pipeline.
* **`tg`** – Telegram bot.
* **`mysql_db`** - MySQL database providing tire models information.
* **`frontend`** - React web-application & Tauri desktop app.
* **`caddy`** - TLS termination & proxy.
* **`tor`** - TOR proxy.

### docker-compose.yaml
```yaml
services:               # Docker compose components       
  tor: ...              # TOR proxy for outgoing traffic
  mysql_db: ...         # MySQL database
  ml: ...               # ML backend
  tg: ...               # Telegram bot
  caddy: ...            # TLS termination & proxy

volumes: ...            # Volumes specification for db and caddy services


secrets:                # Environment variables for the project
  envfile:
    ...

networks:
  proxy: ...            # Proxy network specification
```

### Project layout

The overall project layout:
```bash
Kolobok/
├── Caddyfile           # TLS termination & proxy
├── LICENSE             # MIT License
├── docker-compose.yaml # Docker compose file
├── dataset_utils       # Dataset utils (labeling, synthetic data generation)
├── course_artifacts    # Course artifacts (slides, research, reports, etc.)
├── .env                # Hidden, Required. Project environment
├── ml                  # Backend 
├── frontend            # Frontend
├── mysql_db            # MySQL database
└── tg                  # Telegram bot
```

**Important**:
* **`.env`** file specifies the project environment. For security reasons, not provided in the repository. For more info, see [Configuration](#configuration).

#### ML
`ml` component structure:
```bash
ml/
├── onnx                # Hidden, Required. Directory with exported ONNX models
├── scripts             # Reproducible preprocessing, training, testing, ONNX exporting scripts
├── tests               # CI/CD Tests
├── tire_vision         # Python package with ML/CV utilities 
├── requirements.txt    # Dependencies
├── Dockerfile          # Docker image for the ML backend
├── app.py              # FastAPI app
└── utils.py            # Python module exporting tire_vision for simple integration
```

**Important**:
* **`onnx`** directory must contain exported `.onnx` files with trained models. We do not provide them in the repository, but we provide reproduction code and detailed instructions (see [Reproducibility](#reproducibility)).


#### Frontend
```bash
frontend/
├── kolobok-desktop     # Tauri app
└── web                 # React app
```

#### TG
```bash
tg/
├── Dockerfile          # Docker image for the Telegram bot
└── main.py             # Telegram bot entrypoint
```

#### MySQL
```bash
mysql_db/
├── Dockerfile          # Docker image for the MySQL database
├── init-db.sql         # SQL script to initialize the database
└── models.csv          # Hidden, Required. CSV file
```
**Important**:
* **`models.csv`** file contains tire models database snapshot. Not provided in the repository, requires manual creation and filling. For more info, see [Configuration](#configuration).

#### Course artifacts
```bash
course_artifacts/
├── research                       
│   ├── architecture-plan.md       # ML architecture initial plan
│   ├── ocr_research               # Comparison of OCR models for side-wall information extraction
│   ├── synthetic_dataset_research # Effectiveness of synthetic data generation
│   ├── text_extraction_research   # Research on DB integration in side-wall information extraction pipeline
│   └── thread_depth_research      # Investigation of methods for thread depth estimation
└── userflow_design                # Userflow design in Figma
    ├── figma_link.txt
    └── Userflow.png
```


## Metrics
In this section we provide evaluation details and our finalized metrics for ML component

### Side-Wall Information Extraction
**Evaluation method**: We manually collected and annotated 42 pictures of car wheel side-walls. Each annotation contains:
* Tire brand
* Tire model
* Tire size (format: `<width>/<aspect_ratio><construction_type>/<diameter> <max_weight><speed_rating>` or `<width>/<aspect_ratio><construction_type>/<diameter>`)

We compare *Tire brand* and *Tire model* to the production database and choose specific index of the correct item (if present), or `null` if not present. Tire size is free of dependencies on database.

Then we run Information Extraction pipeline on each picture and take top 5 (by confidence) predictions.

**Metrics**:
* **`recall@1`** - proportion of test predictions with top-1 correct brand / model / size
* **`recall@5`** - proportion of test examples with at least one correct brand / model / size in top-5 predictions
* **`mean_levenshtein_ratio`** - average levenshtein ratio between top-1 prediction and correct brand / model / size

Note: if some of the Ground Truth fields are `null` for specific example, comparison is conducted only for the non-`null` fields.

**Results**:
| Metric | Value |
|--------|-------|
| `recall@1` | 0.81 |
| `recall@5` | 0.90 |
| `mean_levenshtein_ratio` | 0.98 |


### Thread Depth Estimation
**Evaluation details**: We collect 290 images of tires (thread view) from the production dataset with known GT thread depth. For each example, GT thread depth is clipped to the range `[1, 10]` (in test set, no outliers were present).

**Metrics**:
* **`MAE`** - mean absolute error between predicted and GT thread depth
* **`freq_le1`** - proportion of examples with error less than 1mm

**Results**:
| Metric | Value |
|--------|-------|
| `MAE` | 0.51 |
| `freq_le1` | 0.91 |

### Spike Detection & Classification
**Evaluation details**: We collect a set of 81 test images of tires (thread view) with one or more visible spikes. For each image, GT spike location and class are known.

Segmentation model output is arbitrary postprocessed and all connected components of binary mask are transformed into $32 \times 32$ patches. If at least half of the pixels of a spike are present in the patch, this patch is considered as a spike.

After detection, each spike is classified into one of 3 classes: *normal*, *damaged*, *fake* with *fake* representing false positive detections.

Classification evaluation is conducted on a set of 1441 spike crops (from segmentation model detections on test images)

**Metrics**:
* **`average_correct`** - average number of correctly detected spikes
* **`average_miss`** - average number spikes missed by the model
* **`average_incorrect`** - average number of falsely detected spots (no spikes present)
* **`average_ambiguous`** - average number of spikes that were detected by multiple patches
* **`average_error`** - average number of *errors* - missed spikes + false positives
* **`classification_accuracy`** - accuracy of spike class prediction

**Results**:
| Metric | Value |
|--------|-------|
| `average_correct` | 20.18 |
| `average_miss` | 5.10 |
| `average_incorrect` | 2.73 |
| `average_ambiguous` | 0.62 |
| `average_error` | 3.12 |
| `classification_accuracy` | 0.96 |

## Deployment
* Running React application: [link](https://kolobok-meme.vercel.app)
* Backend Swagger documentation: [link](https://kolobok-ml.duckdns.org/docs)
* Tauri desktop application builds: *Coming soon*
* Telegram bot: *Coming soon*

## Reproducibility

We provide comprehensive documentation on how to recreate our project and reproduce our models.

### Dataset
The `dataset_utils` directory contains projects for dataset annotation and synthetic data generation.

#### Label Studio
For tasks such as side-wall unwrapping, tire thread segmentation, spike segmentation, spike classification, we used [Label Studio](https://labelstud.io/) for annotation. To collaboratively annotate the dataset, launch `label-studio` repository using [official instructions](https://github.com/HumanSignal/label-studio) on public domain.

**Example datasets to manually annotate**:
* [Roboflow tire-detect dataset](https://universe.roboflow.com/tiredamage/tire-detect-99ugk)
* [Kaggle tire side profiles dataset](https://www.kaggle.com/datasets/taranmarley/sptire)

#### Synthetic data
We use [Unity](https://unity.com/) game engine to generate scenes with realistic tire models with varying thread depth and configurable backgrounds, lighting, etc. and use [automated tool](./dataset_utils/TireDataset/) to generate images from these scenes.

### Model Training
We use [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://www.pytorchlightning.ai/) frameworks for fine-tuning models from [TorchVision ImageNet collection](https://docs.pytorch.org/vision/main/models.html) and [HuggingFace](https://huggingface.co/).

Training was conducted on `1xA100 GPU` with 80 GB of VRAM.

#### Segmentation (Side-wall unwrapping, Thread segmentation, Spike segmentation)
We use fine-tuned version of [SegFormer](https://github.com/NVlabs/SegFormer) model with pretrained version available at HuggingFace repository `nvidia/segformer-b1-finetuned-ade-512-512`.

Each model was trained using [this script](./ml/scripts/train/train_seg.py) with the following parameters:

* **Side-wall unwrapper**:
  * Total number of train images: 162
  * Number of epochs: 75
  * Validation IoU: 0.91

* **Thread segmentation**:
  * Total number of train images: 1370 + 67 + 80 = 1517 (large pretraining and subsequent fine-tuning)
  * Number of epochs: 5 + 5 + 5 = 15
  * Validation IoU: 0.89

* **Spike segmentation**:
  * Total number of train images: 321
  * Number of epochs: 100
  * Validation IoU: 0.53

For each model, the following hyperparameters were used:
* Learning rate: 0.0001
* Batch size: 64
* Max gradient norm: 1

#### Depth Estimation
We use [this script](./ml/scripts/train/train_depth_regression.py) to train the pre-trained [Swin Transformer](https://github.com/microsoft/Swin-Transformer) model `Swin_V2_T` version with `ImageNet1K` weights.

Hyperparameters:
* Total number of train images: 1370
* Number of epochs: 100
* Batch size: 64
* Learning rate: 0.0001
* Max gradient norm: 1

#### Spike Classification
We fine-tune [GoogLeNet](https://arxiv.org/pdf/1409.4842) model, pretrained on `ImageNet1K` dataset. (training code coming soon)

Hyperparameters:
* Total number of train images: 994
* Number of epochs: 100
* Batch size: 256
* Learning rate: 0.0001
* Max gradient norm: 1

#### Model export
Once the model training is complete, we export our models to ONNX format using scripts from [this directory](./ml/scripts/torch2onnx/)


## Configuration
We provide a [template file](./.env.example) for the project environment.

Place all variables in `.env` (used by docker-compose):

| Variable | Brief Description | Where to Get / How to Set |
|----------|-------------------|----------------------------|
| `API_TOKEN` | Shared secret between bot and API | Generate a secure random string (e.g., `openssl rand -hex 32`) |
| `BOT_TOKEN` | Telegram Bot-Father token | Create a bot via [@BotFather](https://t.me/botfather) on Telegram |
| `ALLOWED_USERS` | Comma-separated list of allowed Telegram user IDs | Get your Telegram user ID (e.g., from [@my_id_bot](https://t.me/my_id_bot)) |
| `OPENROUTER_API_KEY` | OpenRouter API key for LLM inference | Get from [OpenRouter](https://openrouter.ai/keys) |
| `MYSQL_DATABASE` | MySQL database name | Set to your database name (e.g., `kolobok_db`) |
| `MYSQL_ROOT_PASSWORD` | MySQL database root password | Generate a secure password |
| `DB_HOST` | MySQL database hostname | Default: `mysql_db` (Docker service name) |
| `DB_PORT` | MySQL database port | Default: `3306` |

MySQL database credentials are used from backend. You can use DataBase on cloud (recommended) or locally.

## License
MIT (c) Kolobok team