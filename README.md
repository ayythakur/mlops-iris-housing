# MLOps Iris – End-to-End ML Pipeline 🚀

![Build](https://img.shields.io/github/actions/workflow/status/<your-github-username>/<your-repo-name>/ci.yml?branch=main)
![Docker Pulls](https://img.shields.io/docker/pulls/<your-dockerhub-username>/mlops-api)
![Python](https://img.shields.io/badge/python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.x-orange)

A minimal but **complete MLOps workflow** using the Iris dataset:

- **Config-driven** training with scikit-learn + MLflow  
- Best model saved as an sklearn Pipeline (preprocess + model)  
- **FastAPI** service for `/predict` with Pydantic validation  
- **Dockerized** for portable deployment  
- **GitHub Actions** for lint, tests, and Docker image publish  
- Request/response **logging** (+ optional SQLite audit) and `/metrics` (Prometheus)

---

## 📂 Project Structure

.
├─ api/
│ ├─ main.py # FastAPI app: /health, /predict, /metrics
│ └─ schemas.py # Pydantic models
├─ src/
│ ├─ data/ # Data loading & preprocessing
│ ├─ models/ # Training scripts
│ ├─ predict/ # Prediction service
│ └─ utils/ # Logging, MLflow, IO, audit
├─ models/registry/ # Saved best model + metadata
├─ params/iris.yaml # Experiment config
├─ tests/ # Unit tests
├─ Dockerfile
├─ requirements.txt
├─ requirements-dev.txt
└─ .github/workflows/ci.yml


---

## ⚙️ Requirements

- Python **3.11**
- Docker
- GitHub repo (for CI/CD)
- [MLflow](https://mlflow.org/)

---

## 🖥 1) Local Setup

### Create and activate venv
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip

Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

Train the Model
python -m src.models.train --params params/iris.yaml

Run the API Locally
uvicorn api.main:app --host 0.0.0.0 --port 8000

🐳 Docker
Build image:
docker build -t ayythakur/mlops-api:latest .

Run:
docker run --rm -p 8000:8000 <dockerhub-username>/mlops-api:latest


Tests:
pytest -q

