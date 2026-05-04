# Churn Prediction: FIAP PosTech ML

Modelo preditivo de churn para operadora de telecomunicações. Pipeline end-to-end: EDA → MLP PyTorch → API de inferência, com tracking via MLflow e testes automatizados.

---

## Problema e Solução

**Problema:** operadoras perdem receita quando clientes cancelam. Retenção proativa é mais barata que reconquistar clientes perdidos.

**Solução:** MLP (Multilayer Perceptron) treinado com PyTorch que classifica clientes com alta probabilidade de cancelamento, comparada com baselines Scikit-Learn e servida via FastAPI.

**Dataset:** Telco Customer Churn (IBM): 7.043 clientes, 19 features, classificação binária (26,5% churn).

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

## Tech Stack

| Ferramenta | Uso |
|---|---|
| PyTorch | MLP: arquitetura, treino, early stopping |
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

> **Windows:** recomendo o uso do WSL (Windows Subsystem for Linux). Além de ser necessário para os comandos `make`, o WSL oferece um ambiente muito mais compatível com o ecossistema Python/ML em geral: gerenciamento de dependências, scripts de treino e ferramentas de linha de comando funcionam de forma mais confiável do que no PowerShell ou CMD.

```bash
# Criar virtualenv
python -m venv .venv

# Instalar dependências em virtualenv
make install

# Verificar linting
make lint

# Treinar o modelo
make train

# Rodar testes
make test

# Subir a API localmente
make run
```

> **Atenção:** `make run` requer que `make train` tenha sido executado ao menos uma vez. Os artefatos de modelo (`models/`) não são versionados no repositório. Antes de treinar o modelo também não é possível executar todos os testes com `make test`.

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
Cria interações Contract×tenure (monthly_x_tenure, one_year_x_tenure, two_year_x_tenure)
Remove tenure e TotalCharges; StandardScaler (4 numéricas) + OneHotEncoder (14 categóricas) → 39 features
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

| Pasta / Arquivo | Conteúdo |
|---|---|
| `src/churn_nn/` | Código-fonte: ingestão, pré-processamento, arquitetura do MLP e API FastAPI |
| `tests/` | Testes automatizados: schema (pandera), unitários, smoke e API |
| `notebooks/` | EDA, baselines e desenvolvimento da MLP (exploração e análise) |
| `docs/` | ML Canvas, Model Card, arquitetura de deploy e plano de monitoramento |
| `data/raw/` | Dataset original (nunca modificar) |
| `models/` | Artefatos gerados pelo `make train` (não versionados no repositório) |
| `pyproject.toml` | Dependências, configuração do ruff e pytest |
| `Makefile` | Atalhos: `install`, `lint`, `test`, `train`, `run` |

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
  "model_version": "v1.0.0"
}
```

---

## Testes

```bash
make test
```

A suíte cobre quatro camadas:

- **Dados**: valida o schema do CSV bruto (tipos, nulos, domínio de valores) com pandera
- **Pré-processamento**: verifica que o pipeline produz a dimensão de saída correta
- **Modelo**: smoke test confirma que o forward pass da MLP não quebra
- **API**: testa `/health` e `/predict` para respostas válidas e rejeição de inputs inválidos

---

## Documentação

- [`docs/ml-canvas.md`](docs/ml-canvas.md): ML Canvas com stakeholders, métricas de negócio e SLOs
- [`docs/model-card.md`](docs/model-card.md): Model Card com performance, limitações e fairness
- [`docs/deploy-architecture.md`](docs/deploy-architecture.md): arquitetura de deploy com trade-offs
- [`docs/monitoring-plan.md`](docs/monitoring-plan.md): plano de monitoramento e playbook de resposta
