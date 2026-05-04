# Plano de Monitoramento: ChurnMLP

## Objetivo

Garantir que o modelo ChurnMLP mantenha qualidade preditiva e confiabilidade operacional em produção, detectando degradação antes que impacte campanhas de retenção.

---

## Camada 1: Métricas de Modelo

Calculadas mensalmente, após acumular ground truth dos clientes que cancelaram no período (comparar predições do batch anterior com cancelamentos reais).

| Métrica | Baseline (treino) | Alerta Warning | Alerta Crítico |
|---|---|---|---|
| AUC-ROC | 0.848 | queda > 2pp (< 0.828) | queda > 3pp (< 0.818) |
| PR-AUC | 0.633 | queda > 3pp (< 0.603) | queda > 5pp (< 0.583) |
| Recall | 0.886 | queda > 5pp (< 0.836) | queda > 8pp (< 0.806) |
| F1-Score | 0.597 | queda > 5pp (< 0.547) | queda > 9pp (< 0.507) |

**Ação para alerta crítico de AUC-ROC:** retraining imediato obrigatório.

### Justificativa dos thresholds

**AUC-ROC (2pp warning / 3pp crítico):** métrica agnóstica ao threshold de classificação, mede a capacidade discriminativa global do modelo. É estável entre batches e pouco sensível a variações de distribuição de classe. Por isso, qualquer queda pequena é sinal confiável de degradação real; thresholds apertados funcionam como alarme precoce.

**PR-AUC (3pp warning / 5pp crítico):** foca no desempenho na classe positiva (churn), também agnóstica ao threshold. Ligeiramente mais volátil que AUC-ROC em datasets desbalanceados, por isso tem margem um pouco maior.

**Recall (5pp warning / 8pp crítico):** métrica calculada em threshold fixo, portanto mais sensível a variações do perfil do batch. Ainda assim, thresholds apertados são justificados pelo negócio: cada ponto percentual de queda no Recall representa clientes em risco de churn que o modelo deixa de identificar, o erro mais custoso neste contexto.

**F1-Score (5pp warning / 9pp crítico):** combina Recall e Precisão. Uma queda de F1 pode vir de piora na Precisão (mais falsos positivos), que é menos custosa do que perder churners. Por isso admite margem ligeiramente maior que o Recall.

---

## Camada 2: Data Drift

Monitorar a distribuição das features mais preditivas identificadas na EDA. Comparar distribuição do batch de produção com a distribuição do conjunto de treino.

### Features monitoradas

| Feature | Tipo | Método | Threshold de alerta |
|---|---|---|---|
| `Contract` | Categórica | PSI (Population Stability Index) | PSI > 0.2 |
| `tenure` | Numérica | KS-test + PSI | p-value < 0.05 ou PSI > 0.2 |
| `MonthlyCharges` | Numérica | KS-test + PSI | p-value < 0.05 ou PSI > 0.2 |
| `InternetService` | Categórica | PSI | PSI > 0.2 |

**Por que essas features:** são os preditores mais discriminativos identificados na EDA (Cramér's V alto para `Contract` e `InternetService`; correlação -0.35 de `tenure` com churn).

### Output drift

Monitorar a distribuição dos scores de probabilidade ao longo dos batches mensais.

- Calcular: média, mediana, desvio padrão e percentis (p10, p25, p75, p90) dos scores
- Comparar com distribuição do conjunto de validação do treino
- Alerta se a proporção de clientes com score > 0.4 variar > 5pp em relação ao batch anterior

**Importância:** variação na distribuição de scores pode indicar concept drift antes de qualquer degradação observável nas métricas de performance.

---

## Camada 3: Infraestrutura (API Real-Time)

| Métrica | SLO | Alerta Warning | Alerta Crítico |
|---|---|---|---|
| Latência p95 | < 200ms | p95 > 150ms por 5 min | p95 > 200ms por 5 min |
| Taxa de erros HTTP 5xx | < 1% | > 0.5% em 10 min | > 1% em 10 min |
| Disponibilidade | ≥ 99% (horário comercial) | < 99.5% no mês | < 99% acumulado no mês |

---

## Tabela de Alertas e Severidades

| Severidade | Condição | Responsável | Prazo de resposta |
|---|---|---|---|
| **Warning** | Degradação leve de métricas ou drift em features secundárias | Time de Dados | Próximo ciclo mensal |
| **Crítico** | AUC-ROC cai > 3pp ou drift severo em `Contract`/`tenure` | Time de Dados | 48h: iniciar retraining |
| **Incidente** | Pipeline batch falha ou API fora do ar | Responsável técnico | Imediato: investigar e mitigar |

---

## Playbook de Resposta

### Degradação de performance do modelo

1. Verificar se novos segmentos de clientes entraram na base (novo plano, nova região, fusão de operadoras)
2. Rodar análise de drift nas features monitoradas: comparar distribuição do batch atual vs distribuição de treino
3. **Se drift confirmado:** coletar dados novos com ground truth → iniciar retraining com o pipeline padrão (`make train`)
4. **Se sem drift mas performance degradada:** possível concept drift (relação entre features e churn mudou) → revisar features, avaliar nova arquitetura, considerar re-engenharia do modelo
5. Registrar incidente no log com: causa identificada, ação tomada, métricas antes e depois

### Falha de pipeline batch

1. Verificar logs de execução. Erros comuns: schema do CSV diferente, arquivo ausente, incompatibilidade de features no preprocessor
2. **Não reprocessar silenciosamente:** identificar e corrigir a causa raiz antes de reexecutar
3. Após correção: reexecutar `make train` (se necessário) ou só o script de predição batch
4. Documentar: o que falhou, quando, por que e como foi corrigido

### API fora do SLO de latência

1. Verificar carga no servidor: número de requisições simultâneas
2. Verificar se os artefatos de modelo (`mlp_best.pt`, `preprocessor.pkl`) foram corrompidos ou cresceram
3. Se carga elevada: ativar autoscaling ou adicionar réplicas no serviço de hosting
4. Se recorrente: considerar caching do preprocessor ou migração para serving mais eficiente

---

## Critério de Retraining

| Gatilho | Tipo | Ação |
|---|---|---|
| Ciclo mensal | Preventivo | Retraining com dados mais recentes, mesmo sem degradação detectada |
| AUC-ROC cai > 3pp | Reativo | Retraining imediato obrigatório |
| Drift severo em `Contract` ou `tenure` (PSI > 0.2) | Reativo | Investigar + retraining se drift confirmado como real |
| Novos planos ou serviços lançados | Proativo | Retraining com dados que incluam os novos segmentos |

---

## Melhoria Futura

- **Monitoramento automatizado de output drift** com ferramentas como Evidently AI ou Whylogs
- **Análise de fairness por segmento** em cada ciclo de retraining: calcular métricas separadas por `gender` e `SeniorCitizen`
- **Dashboard de monitoramento** com histórico de métricas por batch para visualização de tendências
