# ML Canvas: Predição de Churn em Telecomunicações

## 1. Proposta de Valor

**Problema:** Uma operadora de telecomunicações perde receita recorrente quando clientes cancelam o serviço (churn). A retenção proativa é economicamente superior ao pós-cancelamento, pois age antes da perda da receita futura.

**Solução:** Um modelo preditivo que classifica quais clientes têm alta probabilidade de cancelar, permitindo que a equipe de retenção aja antes do cancelamento com ofertas ou contato proativo.

---

## 2. Stakeholders

| Stakeholder | Papel | Interesse principal |
|---|---|---|
| Diretoria Comercial | Sponsor | Redução do churn rate mensal |
| Time de Retenção (CRM) | Usuário primário | Lista priorizada de clientes em risco |
| Time de Dados / MLOps | Mantenedor | Modelo estável, pipeline reprodutível |
| Clientes | Afetados | Receber ofertas relevantes, não spam |

---

## 3. Dados Disponíveis

**Dataset:** Telco Customer Churn (IBM): 7.043 clientes, 19 features + 1 target

| Grupo | Features | Sinal de churn (EDA) |
|---|---|---|
| Demográfico | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Fraco: `gender` e `Partner` com Cramér's V próximo de 0; `SeniorCitizen` moderado (41,7% vs 24,6%) |
| Contrato | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` | **Forte**: `Contract` é o preditor mais discriminativo do dataset; `tenure` tem correlação de -0,35 com churn |
| Serviços de telefonia | `PhoneService`, `MultipleLines` | Fraco: associação próxima de zero com o target |
| Serviços de internet | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | **Forte**: Fiber Optic com 42% de churn; clientes sem segurança/suporte cancelam ~2x mais |
| Financeiro | `MonthlyCharges`, `TotalCharges` | Moderado: churners pagam ~24% a mais/mês; `TotalCharges` é colinear com `tenure` (r = +0,83) |
| **Target** | `Churn` (Yes/No → 1/0) | 26,5% churn vs 73,5% não-churn |

**Desbalanceamento confirmado:** 26,5% churn (1.869) vs 73,5% não-churn (5.174). Validação cruzada estratificada obrigatória. Accuracy é enganosa: um classificador ingênuo que chuta "não churn" acerta ~73% sem detectar nenhum cancelamento.

**Qualidade dos dados (pós-EDA):**
- `TotalCharges` chegou como `object`; 11 registros em branco (todos com `tenure = 0`) convertido para `float64`, brancos preenchidos com 0
- Nenhum valor ausente, duplicata ou outlier extremo após correção
- `TotalCharges` aproximadamente igual a `tenure × MonthlyCharges` com desvio mediano de 2%: variação esperada por mudanças históricas de plano

**Decisão de feature engineering (guiada pela EDA):**
- `TotalCharges` deve ser removida ou tratada com PCA antes da modelagem: alta colinearidade com `tenure` (r = +0,83) sem sinal independente
- A interação `Contract × tenure` é o padrão mais forte do dataset: clientes com contrato mês a mês e `tenure < 12m` atingem **>51% de taxa de churn**
- Features com associação próxima de zero (`gender`, `PhoneService`) são candidatas a remoção para reduzir ruído

---

## 4. Métricas de Negócio

### Premissas de custo (hipóteses de trabalho, não dados reais)

| Item | Valor estimado |
|---|---|
| Receita mensal média por cliente | R$ 85 |
| Lifetime médio de um cliente retido | 24 meses |
| Custo de uma ação de retenção (ligação + oferta) | R$ 35 |
| Taxa de sucesso da ação de retenção | 30% |

### KPIs de negócio

**Receita protegida por mês:**
> Clientes retidos com sucesso × Receita mensal média × Lifetime restante estimado

**Custo-benefício da campanha de retenção:**
> Receita protegida / (Nº de clientes abordados × R$ 35)

**Meta:** razão custo-benefício maior ou igual a 3:1, ou seja, para cada R$ 1 gasto em retenção, recuperar ao menos R$ 3 em receita.

### Impacto de erros

| Tipo de erro | Impacto |
|---|---|
| **Falso negativo** (churn não detectado) | Cliente cancela sem abordagem: perda total da receita futura |
| **Falso positivo** (não-churn abordado) | Custo da ação de retenção desnecessária (R$ 35) + possível atrito |

**Conclusão:** falsos negativos são mais custosos que falsos positivos neste contexto. O modelo deve priorizar **recall** sobre precisão. O risco é especialmente alto no segmento crítico identificado na EDA: clientes com contrato mês a mês, `tenure < 12m` e `MonthlyCharges` elevado (Fiber Optic).

---

## 5. Métricas Técnicas

| Métrica | Meta mínima | Justificativa |
|---|---|---|
| **AUC-ROC** | >= 0.80 | Discriminação geral, independente do threshold |
| **PR-AUC** | >= 0.60 | Prioritária com classes desbalanceadas: insensível ao volume de verdadeiros negativos, que aqui é grande |
| **Recall (classe churn)** | >= 0.70 | Capturar a maioria dos churns reais; falsos negativos têm custo maior |
| **F1-Score (classe churn)** | >= 0.60 | Equilíbrio entre precisão e recall |
| **Precisão (classe churn)** | >= 0.55 | Evitar volume excessivo de falsos positivos |

**Threshold padrão:** 0.4 (rebaixado de 0.5), favorecendo recall dada a assimetria de custo confirmada pela EDA.

**Nota:** accuracy não é uma métrica válida de avaliação neste problema. Um classificador constante "não churn" alcançaria ~73% de accuracy sem qualquer utilidade preditiva.

---

## 6. SLOs: Service Level Objectives

| Dimensão | SLO |
|---|---|
| Latência de inferência (API) | p95 < 200 ms por requisição |
| Disponibilidade da API | >= 99% em horário comercial |
| Freshness do modelo | Retraining mensal ou quando AUC-ROC cair > 3pp em produção |
| Cobertura de dados | Pipeline deve processar 100% das features sem falha silenciosa |

---

## 7. Saídas do Modelo

**Modo batch (principal):** CSV com `customerID`, `churn_probability`, `churn_prediction` (0/1), executado mensalmente.

**Modo real-time (API):** endpoint `/predict` recebe features de um cliente e retorna probabilidade + predição para uso em sistemas CRM.

---

## 8. Riscos e Limitações

| Risco | Severidade | Mitigação |
|---|---|---|
| Dataset estático (sem drift temporal real) | Alta | Monitorar distribuição de features em produção |
| Viés de representatividade (dataset IBM, EUA) | Média | Documentar no Model Card; validar com dados locais |
| Clientes novos (`tenure = 0`): apenas 11 registros (0,15% da base) | Baixa *(revisado)* | Preencher `TotalCharges` com 0; sem tratamento especial necessário dado o volume mínimo |
| Mudança de regra de negócio (novos planos/serviços) | Alta | Retraining periódico obrigatório |
| Ação de retenção gera atrito se mal direcionada | Baixa | Limitar abordagem a clientes com score >= threshold |
| Multicolinearidade: `TotalCharges` e `tenure` (r = +0,83) | Média | Remover `TotalCharges` do modelo ou aplicar PCA; não usar ambas as features simultaneamente |
| Segmento Fiber Optic com 42% de churn pode introduzir viés | Média | Monitorar performance separada por segmento de internet em produção |

---

## 9. Critérios de Go/No-Go para Produção

- [x] AUC-ROC >= 0.80 no conjunto de teste (**atingido: 0.848**)
- [x] PR-AUC >= 0.60 no conjunto de teste (**atingido: 0.633**)
- [x] Recall (churn) >= 0.70 com threshold = 0.4 (**atingido: 0.886**)
- [x] Todos os testes automatizados passando (`pytest`)
- [x] API respondendo em < 200 ms no p95
- [x] Model Card preenchido com limitações documentadas

**Observação sobre precisão:** a meta de precisão >= 0.55 não foi atingida (resultado: 0.450). O desvio é consequência direta do threshold=0.4, escolhido deliberadamente para priorizar recall dado o custo assimétrico dos erros: falsos negativos (churns não detectados) implicam perda total da receita futura, enquanto falsos positivos custam apenas R$35 por ação de retenção desnecessária. O trade-off é consciente, documentado no Model Card e validado pelas premissas de custo da Seção 4.

---

## 10. Principais Achados da EDA

Resumo dos padrões descobertos na análise exploratória que informam diretamente as decisões de modelagem:

| Hipótese | Resultado | Evidência-chave |
|---|---|---|
| H1: Contrato mensal → mais churn | Confirmada | 43% vs 11% (anual) vs 3% (bienal); `Contract` tem o maior Cramér's V do dataset |
| H2: Tenure alto → menos churn | Confirmada | Correlação -0,35; taxa cai de 47,7% (0-12m) para 9,5% (>4 anos) |
| H3: Fiber Optic → mais churn | Confirmada | 42% vs 19% (DSL) vs 7% (sem internet); serviço mais caro da base |
| H4: Sem serviços de proteção/suporte → mais churn | Confirmada | ~2x mais churn; `OnlineSecurity` e `TechSupport` são os mais discriminativos |
| H5: Idosos → mais churn | Confirmada | 41,7% vs 24,6%; impacto absoluto limitado (~16% da base) |
| H6: `MonthlyCharges` alto → mais churn | Confirmada | Mediana $79,65 (churnou) vs $64,43 (ficou); diferença de ~24% |

**Padrão dominante:** clientes com contrato mês a mês e `tenure < 12m` ultrapassam **51% de taxa de churn**. A combinação dessas duas features é o sinal mais forte e deve ser considerada como feature de interação no pipeline.
