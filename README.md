# UCI Heart Disease Data Augmentation using Gemma 4 and LightRAG

A pipeline for generating synthetic patient records by combining retrieval-augmented generation (LightRAG) with a locally-hosted Gemma 4 language model. Synthetic records are generated from seed patient profiles grounded in clinical guidelines, then used to augment the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) for downstream classification tasks.

---

## How It Works

```
Medical Guidelines (.txt)
        │
        ▼
   LightRAG Index          ←─── Knowledge graph + vector store
        │
        │  hybrid retrieval (local + global)
        ▼
  Clinical Constraints     ←─── e.g. BP/chol ranges for age/sex profile
        │
        ├── Seed Patient Profile
        ▼
   Gemma 4 (Ollama)        ←─── Generates one complete patient record
        │
        ▼
  Synthetic CSV dataset
        │
        ▼
  Classifier Evaluation    ←─── Compare vs. original-only baseline
```

The key idea is that the LLM is grounded in real medical knowledge retrieved from clinical guidelines — producing records where feature correlations (age, cholesterol, blood pressure, etc.) are medically coherent.

---

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- [Ollama](https://ollama.com/) installed and running
- Gemma 4 model pulled: `ollama pull gemma4:e4b`
- `nomic-embed-text` pulled: `ollama pull nomic-embed-text`

> **Remote / Kaggle setup:** If running Ollama on a remote machine exposed via [ngrok](https://ngrok.com/), set `OLLAMA_HOST` to the tunnel URL (see [Environment Variables](#environment-variables)). The pipeline includes automatic tunnel health checks and retry logic.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Phues/UCI-Heart-Disease-Data-Augmentation-using-Gemma4-and-LightRAG.git
cd UCI-Heart-Disease-Data-Augmentation-using-Gemma4-and-LightRAG
```

### 2. Create and activate the Conda environment

```bash
conda create -n heartrag python=3.11 -y
conda activate heartrag
```

### 3. Install Python dependencies

```bash
pip install lightrag-hku ollama pandas numpy scikit-learn \
            matplotlib seaborn jupyter notebook \
            python-dotenv tqdm
```

### 4. Patch LightRAG embedding dimension

LightRAG's Ollama backend defaults to `embedding_dim=1024`, but `nomic-embed-text` produces 768-dimensional embeddings. Apply the fix after installation:

```bash
sed -i 's/embedding_dim=1024/embedding_dim=768/g' \
    ~/miniconda3/envs/heartrag/lib/python3.11/site-packages/lightrag/llm/ollama.py
```

> **Note:** Replace `heartrag` with your chosen environment name if different. On macOS, use `sed -i ''` instead of `sed -i`.

Verify the patch applied:

```bash
grep "embedding_dim" \
    ~/miniconda3/envs/heartrag/lib/python3.11/site-packages/lightrag/llm/ollama.py
# Should output: embedding_dim=768
```

### 5. Pull Ollama models

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text
```

---

## Environment Variables

Create a `.env` file in the project root (or export directly in your shell):

```bash
# Local Ollama (default — no changes needed)
OLLAMA_HOST=http://localhost:11434

# Remote Ollama via ngrok tunnel (Kaggle / Colab setup)
# OLLAMA_HOST=https://your-tunnel-id.ngrok-free.app
```

The pipeline reads `OLLAMA_HOST` dynamically on every request, so you can update the tunnel URL mid-run without restarting.

---

## Usage

### Step 1 — Ingest medical guidelines into LightRAG

Place your clinical guideline text files in `rag/guidelines/`. Then run:

```bash
cd src
python rag_setup.py
```

This builds the knowledge graph index in `rag/index/` and runs a test query to confirm retrieval is working. Ingested files are checkpointed — re-running skips already-processed files.

### Step 2 — Generate synthetic records

```bash
python pipeline.py

python pipeline.py 200
```

Output is saved to `data/processed/synthetic_heart_local.csv`. 

### Step 3 — Generate a single record (quick test)

```bash
python generator.py
```

Uses a hardcoded seed profile (`age=55, sex=male, chol=240`) and prints the generated JSON record.

### Step 4 — Run the classification notebook

```bash
jupyter notebook notebooks/heart_disease_classification.ipynb
```

The notebook evaluates classifiers on the original dataset and on the augmented (original + synthetic) dataset for comparison.

---

## Dataset

The UCI Heart Disease dataset contains 303 patient records with 13 features and a binary target label (0 = no disease, 1 = disease present).

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 0 = female, 1 = male |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mmHg) |
| `chol` | Serum cholesterol (mg/dL) |
| `fbs` | Fasting blood sugar > 120 mg/dL |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) |
| `thal` | Thalassemia type (1–3) |
| `target` | Heart disease presence (0/1) |

---