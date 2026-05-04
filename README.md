# Churn Prediction — FIAP PosTech ML

Modelo preditivo de churn para operadora de telecomunicações. Pipeline end-to-end: EDA → MLP PyTorch → API de inferência, com tracking via MLflow e testes automatizados.

---

## Problema e Solução

**Problema:** operadoras perdem receita quando clientes cancelam. Retenção proativa é mais barata que reconquistar clientes perdidos.

**Solução:** MLP (Multilayer Perceptron) treinado com PyTorch que classifica clientes com alta probabilidade de cancelamento, comparada com baselines Scikit-Learn e servida via FastAPI.

**Dataset:** Telco Customer Churn (IBM) — 7.043 clientes, 19 features, classificação binária (26,5% churn).

---

## Métricas do Modelo (conjunto de teste, threshold=0.4)

| Métrica | Meta | Resultado |
|---|---|---|
| AUC-ROC | ≥ 0.80 | **0.848** ✅ |
| PR-AUC | ≥ 0.60 | **0.633** ✅ |
| Recall | ≥ 0.70 | **0.886** ✅ |
| F1-Score | ≥ 0.60 | **0.597** ⚠️ |
| Precisão | ≥ 0.55 | **0.450** |

Threshold rebaixado para 0.4 para priorizar recall: falsos negativos (churns não detectados) têm custo maior que falsos positivos neste domínio.

---

## Stack Técnica

| Ferramenta | Uso |
|---|---|
| PyTorch | MLP — arquitetura, treino, early stopping |
| Scikit-Learn | Pipeline de pré-processamento + baselines |
| MLflow | Tracking de experimentos (params, métricas, artefatos) |
| FastAPI | API de inferência |
| pytest | Testes automatizados (smoke, schema, API) |
| ruff | Linting |
| pandera | Validação de schema dos dados |
| pydantic | Validação dos requests da API |

---

## Setup

**Pré-requisitos:** Python 3.11+, make

```bash
# Instalar dependências em virtualenv
make install

# Verificar linting
make lint

# Treinar o modelo (obrigatório antes de rodar a API ou os testes)
make train

# Rodar testes
make test

# Subir a API localmente
make run
```

> **Atenção:** `make run` e `make test` requerem que `make train` tenha sido executado ao menos uma vez. Os artefatos de modelo (`models/`) não são versionados no repositório.

> **Windows:** os comandos `make` requerem WSL.

---

## Arquitetura

```
data/raw/telco-churn.csv
    │
    ▼
load_data()                    ← src/churn_nn/data/preprocessing.py
Corrige TotalCharges, converte Churn → 0/1
    │
    ▼
build_preprocessor()           ← src/churn_nn/data/preprocessing.py
StandardScaler (numéricas) + OneHotEncoder (categóricas)
    │
    ├── fit → models/preprocessor.pkl
    │
    ▼
ChurnMLP(input_dim=39)         ← src/churn_nn/models/mlp.py
Linear(39→64) → ReLU → Dropout(0.3)
Linear(64→32) → ReLU → Dropout(0.3)
Linear(32→1)
    │
    ├── best weights → models/mlp_best.pt
    │
    ▼
API FastAPI                    ← src/churn_nn/api/app.py
POST /predict → probabilidade + predição
GET  /health  → status + model_version
```

---

## Estrutura do Repositório

```
.
├── src/churn_nn/
│   ├── config.py              # Hiperparâmetros e paths centralizados
│   ├── train.py               # Script de treino (make train)
│   ├── data/
│   │   └── preprocessing.py   # load_data(), build_preprocessor()
│   ├── models/
│   │   └── mlp.py             # class ChurnMLP(nn.Module)
│   └── api/
│       ├── app.py             # FastAPI: /health, /predict, middleware
│       └── schemas.py         # Pydantic: CustomerFeatures, PredictionResponse
├── tests/
│   ├── conftest.py            # Fixtures compartilhadas
│   ├── test_schema.py         # Pandera: valida schema do CSV bruto
│   ├── test_preprocessing.py  # Unitário: preprocessor fit/transform
│   ├── test_smoke.py          # Smoke: ChurnMLP.forward() não quebra
│   └── test_api.py            # API: /health 200, /predict 200 e 422
├── notebooks/
│   ├── 01-eda.ipynb           # Análise exploratória completa
│   ├── 02-baselines.ipynb     # DummyClassifier + Regressão Logística
│   └── 03-mlp.ipynb           # Desenvolvimento e avaliação da MLP
├── docs/
│   ├── ml-canvas.md           # ML Canvas: stakeholders, métricas, SLOs
│   ├── model-card.md          # Model Card: performance, limitações, fairness
│   ├── deploy-architecture.md # Arquitetura de deploy: batch vs real-time
│   └── monitoring-plan.md     # Plano de monitoramento e playbook
├── data/
│   └── raw/telco-churn.csv    # Dataset original (nunca modificar)
├── models/                    # Artefatos gerados por make train (não versionados)
│   ├── mlp_best.pt
│   ├── preprocessor.pkl
│   └── model_metadata.json
├── pyproject.toml             # Dependências, ruff, pytest
└── Makefile                   # install, lint, test, train, run
```

---

## MLflow

Experimentos rastreados em SQLite local (`mlruns.db`), gerado automaticamente pelo `make train`.

```bash
# Ativar ambiente e abrir UI
source .venv/bin/activate
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Acesse [http://localhost:5000](http://localhost:5000). O experimento aparece como **telco-churn**.

Cada run registra: hiperparâmetros, métricas do teste, artefatos do modelo e hash do commit Git.

---

## API

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status da API e versão do modelo |
| POST | `/predict` | Predição de churn para um cliente |

### Exemplo: POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 2,
    "MonthlyCharges": 79.85,
    "TotalCharges": 159.70,
    "SeniorCitizen": 0,
    "gender": "Male",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
  }'
```

**Resposta:**
```json
{
  "churn": true,
  "probability": 0.8312,
  "threshold": 0.4,
  "model_version": "ca147906"
}
```

---

## Testes

```bash
make test
```

| Suite | O que verifica |
|---|---|
| `test_schema.py` | Schema do CSV bruto com pandera (tipos, nulls, domínio de valores) |
| `test_preprocessing.py` | `build_preprocessor()` fit/transform sem erros; dimensão de saída correta |
| `test_smoke.py` | `ChurnMLP.forward()` retorna tensor com shape correto |
| `test_api.py` | `/health` retorna 200; `/predict` válido retorna 200; inválido retorna 422 |

---

## Documentação

- [`docs/ml-canvas.md`](docs/ml-canvas.md) — ML Canvas: stakeholders, métricas de negócio, SLOs
- [`docs/model-card.md`](docs/model-card.md) — Model Card: performance, limitações, fairness
- [`docs/deploy-architecture.md`](docs/deploy-architecture.md) — Arquitetura de deploy com trade-offs
- [`docs/monitoring-plan.md`](docs/monitoring-plan.md) — Plano de monitoramento e playbook de resposta
