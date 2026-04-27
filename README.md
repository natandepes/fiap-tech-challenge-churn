# Churn Prediction — FIAP PosTech ML

Modelo preditivo de churn para operadora de telecomunicações, com pipeline end-to-end do EDA até API de inferência.

## Visão Geral

Este projeto treina uma rede neural MLP (PyTorch) para classificar clientes com risco de cancelamento, comparada com baselines Scikit-Learn, rastreada com MLflow e servida via FastAPI.

## Setup

```bash
# Instalar dependências
pip install -e ".[dev]"

# Verificar instalação
make test
```

## Estrutura

```
src/        Código-fonte modularizado
data/       Dados raw e processados
models/     Artefatos dos modelos treinados
tests/      Testes automatizados
notebooks/  EDA e exploração
docs/       Model Card e documentação
```

## Comandos

```bash
make lint     # Linting com ruff
make test     # Rodar testes (pytest)
make train    # Treinar o modelo
make run      # Subir a API localmente
```

## Dataset

Telco Customer Churn (IBM) — ~7.000 clientes, ~20 features, classificação binária.

## Stack

- **PyTorch** — MLP
- **Scikit-Learn** — pipelines de pré-processamento e baselines
- **MLflow** — tracking de experimentos
- **FastAPI** — API de inferência
