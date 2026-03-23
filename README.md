# 🚀 DeepClassifier: Advanced MLOps for Arabic Text Classification

[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org/)
[![BERT](https://img.shields.io/badge/Model-BERT--Arabic-orange)](https://huggingface.co/models)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**DeepClassifier** is a production-ready MLOps framework designed for classifying Arabic questions by **complexity** and **intent**. It leverages state-of-the-art BERT models while maintaining a highly decoupled, modular architecture suitable for enterprise-grade machine learning pipelines.

---

## 🏗️ Architecture & Design Patterns

The project is built on **Production-Grade MLOps Engineering**, ensuring that logic, configuration, and operations are strictly separated.

### 💉 Separation of Concerns (SoC)
- **Data Layer**: Implements a **Strategy Pattern** for preprocessing. This allows adding new cleaning steps without modifying the core pipeline logic.
- **Model Layer**: Features a customized BERT-based Classifier with a dedicated `Trainer` class that abstracts the training loops and validation logic.
- **Pipeline Layer**: Modular scripts for each stage (Preprocessing → Training → Evaluation) ensure that failures in one stage are isolated.

### ⚙️ MLOps Core Features
- **Experiment Tracking**: Full integration with **MLflow** to log hyperparameters, training metrics, and model artifacts.
- **Reproducibility**: All project parameters are centralized in `config/main_config.yaml`.
- **Automation**: A robust PowerShell runner (`run_pipeline.ps1`) automates the entire staging process.
- **Versioning**: Systematic storage of dataset versions and model checkpoints in the `experiments/` directory.

---

## 📁 Project Structure

```text
MLOps-Quesion-DeepClassfier/
├── config/                 # YAML Configuration files
├── experiments/            # Versioned datasets and model checkpoints
├── mlruns/                 # MLflow tracking data (Local DB)
├── src/                    # Source Code
│   ├── data/               # Data loading, validation, and strategy-based preprocessing
│   ├── features/           # Feature engineering and custom dataset loaders
│   ├── models/             # Model architectures and specialized Trainer
│   └── pipelines/          # Entry-point scripts for each pipeline stage
├── run_pipeline.ps1        # Automated pipeline runner (PowerShell)
└── requirements.txt        # Project dependencies
```

---

## 🚀 Quick Start

### 1. Environment Setup
Ensure you have [Conda](https://docs.conda.io/en/latest/) installed, then create and activate the project environment:

```bash
conda create -n MLOps python=3.10
conda activate MLOps
pip install -r requirements.txt
```

### 2. Configure the Pipeline
All settings (paths, hyperparams, MLflow URI) are located in `config/main_config.yaml`. Review it before running the project.

### 3. Run the Automated Pipeline
You can run the full pipeline or individual stages using the provided runner:

```powershell
# Run the entire pipeline
.\run_pipeline.ps1 -Stage all

# Run specific stages
.\run_pipeline.ps1 -Stage preprocess
.\run_pipeline.ps1 -Stage train
.\run_pipeline.ps1 -Stage eval
```

---

## 📊 Monitoring & Tracking

Launch the MLflow UI to visualize your experiments and compare model versions:

```bash
# Note: Ensure your tracking URI in main_config.yaml matches
mlflow ui
```

---

## 🛠️ Built With

- **NLP**: [Transformers](https://huggingface.co/docs/transformers/index) (BERT)
- **Deep Learning**: [PyTorch](https://pytorch.org/)
- **MLOps**: [MLflow](https://mlflow.org/)
- **Data Handling**: [Pandas](https://pandas.pydata.org/), [Scikit-learn](https://scikit-learn.org/)
