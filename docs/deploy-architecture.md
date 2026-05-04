# Arquitetura de Deploy — Churn Prediction

## Contexto de Decisão

O modelo ChurnMLP é usado para **retenção proativa de clientes**: a equipe de CRM recebe uma lista de clientes com alto risco de churn e executa campanhas de contato. A arquitetura de deploy deve servir esse processo da forma mais eficiente e confiável possível.

Dois modos são possíveis: **batch** (processamento periódico de toda a base) e **real-time** (resposta por requisição sob demanda via API). Cada um tem características diferentes em custo, complexidade e adequação ao caso de uso.

---

## Comparativo: Batch vs Real-Time

| Dimensão | Batch | Real-Time (API) |
|---|---|---|
| **Latência** | Alta (minutos a horas por execução) | Baixa (< 200ms por requisição) |
| **Throughput** | Alto (toda a base de uma vez) | Baixo a médio (depende de carga) |
| **Custo operacional** | Baixo (executa sob demanda) | Alto (infraestrutura permanente 24/7) |
| **Complexidade de manutenção** | Baixa (script + cron) | Alta (serving, autoscaling, SLOs) |
| **Debugging** | Simples (saída é arquivo auditável) | Complexo (erros em tempo real, logs distribuídos) |
| **Alinhamento ao caso de uso** | Alto (campanhas são mensais) | Médio (útil para consultas pontuais) |
| **Dependência de disponibilidade** | Nenhuma durante execução | Contínua (falha = indisponibilidade) |
| **Resultado defasado?** | Sim (predições têm data fixa) | Não (estado atual do cliente) |

---

## Recomendação: Batch como Modo Principal

O processo de retenção de clientes é orientado a **campanhas periódicas**: a equipe de CRM planeja ações mensais baseadas em uma lista priorizada, não toma decisões individuais em tempo real por cliente.

**Por que batch é a escolha certa aqui:**

1. **Alinhamento com o ciclo de negócio:** campanhas de retenção são planejadas com antecedência e executadas em blocos. Não há benefício operacional em ter predições sub-segundo quando a ação vai acontecer em dias.

2. **Custo:** o batch roda uma vez por mês, sem infraestrutura permanente. A API exigiria um container ativo 24/7, com custo fixo mesmo em períodos sem uso.

3. **Simplicidade e confiabilidade:** um script Python agendado tem menos pontos de falha que uma API com autoscaling, balanceamento de carga e SLOs de latência. Para um modelo que roda mensalmente, essa complexidade não se justifica.

4. **Auditabilidade:** o output do batch é um arquivo CSV versionável, com todas as predições rastreáveis. Erros são detectáveis e corrigíveis offline antes de qualquer ação.

**Quando o real-time faz sentido (complementar):**

A API FastAPI já implementada serve casos pontuais: um agente de CRM que quer consultar o score de um cliente específico durante uma ligação, ou um sistema de automação que precisa de predição ao cadastrar um novo contrato. Esses são usos secundários — úteis, mas não o fluxo principal.

---

## Fluxo Batch (Principal)

```
[Fonte de dados]
    CSV mensal atualizado: data/raw/telco-churn.csv
         │
         ▼
[Pipeline de Dados]
    load_data()  ← src/churn_nn/data/preprocessing.py
    Correção de TotalCharges, conversão de Churn → 0/1
         │
         ▼
[Pré-processamento]
    preprocessor.pkl  (Pipeline sklearn)
    Interações Contract×tenure + StandardScaler + OneHotEncoder → 39 features
         │
         ▼
[Inferência]
    mlp_best.pt  (ChurnMLP PyTorch)
    threshold = 0.4
         │
         ▼
[Saída]
    CSV: customerID | churn_probability | churn_prediction (0/1)
         │
         ▼
[Consumo]
    Sistema CRM: importa lista, prioriza por probabilidade
    Equipe de retenção: executa campanha de contato
```

---

## Fluxo Real-Time (Complementar)

```
[Sistema CRM / Frontend]
    Trigger: agente em atendimento ou novo cadastro
         │
         ▼
[API FastAPI]  ← src/churn_nn/api/app.py
    POST /predict
    Validação Pydantic (CustomerFeatures)
    Middleware de latência (log p95)
         │
         ▼
[Pré-processamento]
    preprocessor.pkl (carregado no startup da API)
         │
         ▼
[Inferência]
    mlp_best.pt (carregado no startup da API)
    torch.no_grad() → sigmoid → threshold 0.4
         │
         ▼
[Resposta JSON]
    {
      "churn": true/false,
      "probability": 0.73,
      "threshold": 0.4,
      "model_version": "ca147906"
    }
```

---

## Proposta de Infraestrutura (Sem Implementação)

### Batch

| Componente | Opção simples | Opção robusta |
|---|---|---|
| Agendamento | cron (Linux/servidor) | Apache Airflow DAG |
| Execução | `python -m churn_nn.train` (com modo predict) | Container Docker + orquestrador |
| Storage de saída | Sistema de arquivos local ou NFS | Amazon S3 / Google Cloud Storage |
| Monitoramento | Log de execução + alertas por e-mail | Airflow + Slack/PagerDuty |

### Real-Time (API)

| Componente | Opção |
|---|---|
| Container | Docker com `uvicorn churn_nn.api.app:app --port 8000` |
| Serving | Cloud Run (GCP), Railway, ou ECS Fargate (AWS) |
| Health check | `GET /health` → `{"status": "ok", "model_version": "..."}` |
| Escalabilidade | Autoscaling por CPU/requisições no serviço gerenciado |
| Segredos | Variáveis de ambiente para `MODELS_DIR` |
