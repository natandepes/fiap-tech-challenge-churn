# Model Card: ChurnMLP

## Identificação

| Campo | Valor |
|---|---|
| **Nome** | ChurnMLP |
| **Versão** | v1.0.0 |
| **Data de treino** | 2026-05-04 |
| **Desenvolvedor** | Natan Depes, FIAP PosTech ML (2026) |
| **Licença** | Uso acadêmico |
| **Repositório** | Tech Challenge Fase 1: Churn NN |

---

## Arquitetura

**Tipo:** Multilayer Perceptron (MLP), PyTorch

**Topologia:**
```
Entrada (39 features) → Linear(39→64) → ReLU → Dropout(0.3)
                      → Linear(64→32) → ReLU → Dropout(0.3)
                      → Linear(32→1)  → Sigmoid (inferência)
```

**Hiperparâmetros de treino:**

| Parâmetro | Valor |
|---|---|
| Loss | BCEWithLogitsLoss com pos_weight (balanceamento de classes) |
| Otimizador | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Max epochs | 100 |
| Early stopping (patience) | 10 épocas sem melhora no val_loss |
| Épocas efetivas | 18 (parada precoce) |
| Threshold de decisão | 0.4 (rebaixado de 0.5 para priorizar recall) |
| Seed | 42 (torch, numpy, random, sklearn) |

---

## Uso Pretendido

**Contexto:** campanhas mensais de retenção proativa de clientes de telecomunicações.

**Usuários:** time de CRM/retenção, plataformas de automação de marketing.

**Modos de operação:**
- **Batch (principal):** geração mensal de lista priorizada de clientes com risco de churn
- **Real-time (complementar):** endpoint `/predict` para consulta pontual durante atendimento

**Fora do escopo:**
- Decisões de crédito ou seguros
- Qualquer contexto que exija auditoria formal de fairness
- Populações fora do perfil do dataset (telecomunicações, mercado norte-americano)

---

## Performance no Conjunto de Teste

**Conjunto de teste:** 1.057 registros (15% estratificado do dataset original).  
**Threshold:** 0.4

| Métrica | Meta (ML Canvas) | Resultado | Status |
|---|---|---|---|
| AUC-ROC | ≥ 0.80 | **0.848** | ✅ |
| PR-AUC | ≥ 0.60 | **0.633** | ✅ |
| Recall (churn) | ≥ 0.70 | **0.886** | ✅ |
| F1-Score (churn) | ≥ 0.60 | **0.597** | ⚠️ marginal |
| Precisão (churn) | ≥ 0.55 | **0.450** | ❌ abaixo da meta |

**Nota sobre precisão:** a precisão abaixo da meta é consequência direta do threshold=0.4, escolhido deliberadamente para maximizar recall. Neste domínio, falsos negativos (churns não detectados) têm custo maior que falsos positivos (abordagens desnecessárias): perder um cliente gera perda total da receita futura, enquanto abordar um cliente que não cancelaria custa apenas R$35 de ação de retenção. O trade-off é consciente e documentado no ML Canvas.

---

## Atributos Sensíveis

Os seguintes atributos demograficamente sensíveis estão presentes no dataset e foram utilizados como features de entrada:

| Atributo | Tipo | Observação |
|---|---|---|
| `gender` | Categórico (Male/Female) | Cramér's V ≈ 0 com churn (EDA); baixo poder preditivo |
| `SeniorCitizen` | Binário (0/1) | Cramér's V moderado; taxa de churn de 41,7% entre idosos vs 24,6% entre não-idosos |
| `Partner` | Categórico (Yes/No) | Cramér's V próximo de 0 (EDA) |
| `Dependents` | Categórico (Yes/No) | Correlação fraca com churn |

**Análise de fairness por grupo: não realizada.**

Não foi conduzida análise de disparate impact, equalização de odds ou qualquer métrica de fairness por subgrupo demográfico. Esta é uma limitação conhecida e declarada. Antes de uso em produção real, recomenda-se:
1. Calcular métricas de performance separadas por `gender` e `SeniorCitizen`
2. Verificar se recall e precisão são equivalentes entre grupos
3. Avaliar se as predições geram impactos desproporcionais em algum segmento

---

## Dados de Treinamento

**Dataset:** Telco Customer Churn (IBM Sample Data)  
**Volume:** 7.043 registros, 19 features + 1 target  
**Desbalanceamento:** 26,5% churn (1.869) vs 73,5% não-churn (5.174)

**Split (estratificado, seed=42):**
- Treino: 4.929 registros (70%)
- Validação: 1.057 registros (15%)
- Teste: 1.057 registros (15%)

**Pré-processamento:**
- `TotalCharges`: convertida de object para float64 (API aceita o campo); **removida do modelo** (r=0.83 com `tenure`, alta multicolinearidade)
- `tenure`: **não entra diretamente** no modelo; substituída por 3 termos de interação Contract×tenure
- `gender`, `PhoneService`: removidas (Cramér's V ≈ 0 com target na EDA)
- `customerID`: removida (identificador, sem sinal preditivo)
- **Termos de interação criados:** `monthly_x_tenure`, `one_year_x_tenure`, `two_year_x_tenure`
- Numéricas (`MonthlyCharges` + 3 interações): StandardScaler → 4 features
- Categóricas (14 variáveis): OneHotEncoder com `drop="if_binary"` e `sparse_output=False` → 35 features
- **Input total ao modelo: 39 features**

---

## Limitações e Vieses Conhecidos

| Limitação | Severidade | Detalhe |
|---|---|---|
| **Dataset estático** | Alta | Sem drift temporal real; modelo pode degradar com mudanças de mercado, novos planos ou comportamentos emergentes |
| **Representatividade geográfica** | Média | Dataset IBM/EUA; comportamento de churn pode diferir significativamente em mercado brasileiro |
| **Viés Fiber Optic** | Média | 42% de taxa de churn nesse segmento vs 19% (DSL); modelo pode ser sistematicamente mais agressivo para clientes de fibra óptica |
| **Fairness não auditada** | Média | Possível disparidade de performance por grupo demográfico (`gender`, `SeniorCitizen`) não investigada |
| **Precisão abaixo da meta** | Baixa | 0.450 vs meta ≥ 0.55; consequência do threshold=0.4; aceitável dado o trade-off de custo documentado |
| **Clientes novos (tenure=0)** | Baixa | Apenas 11 registros no treino; predições menos confiáveis para clientes recém-adquiridos |
| **Multicolinearidade resolvida** | Baixa | `TotalCharges` removida (r=+0.83 com `tenure`); `tenure` substituída por 3 termos de interação Contract×tenure |

---

## Cenários de Falha

| Cenário | Causa | Consequência esperada |
|---|---|---|
| **Cliente com tenure=0** | Subrrepresentação no treino (11 registros) | Probabilidade menos calibrada; pode subestimar ou superestimar churn |
| **Novo plano ou serviço sem histórico** | Feature fora da distribuição de treino | Encoders produzem valores fora do range observado; predição degradada |
| **Drift temporal prolongado** | Modelo treinado em snapshot estático | AUC-ROC degrada gradualmente sem retraining; alertas do plano de monitoramento devem detectar |
| **Mudança de regra de negócio** | Novos contratos ou estrutura de preços | Features como `Contract` e `MonthlyCharges` perdem poder preditivo; retraining obrigatório |
| **Arquivo CSV com schema diferente** | Mudança de fonte de dados | Pipeline falha com erro de schema (pandera valida) ou produz predições incorretas silenciosamente |

---

## Critérios de Go/No-Go para Produção

| Critério | Limiar | Resultado |
|---|---|---|
| AUC-ROC | ≥ 0.80 | 0.848 |
| PR-AUC | ≥ 0.60 | 0.633 |
| Recall (churn, threshold = 0.4) | ≥ 0.70 | 0.886 |
| Testes automatizados (`pytest`) | 100% passando | 12/12 |
| Latência p95 | < 200ms | 15ms |

**Obs.:** latência medida via FastAPI TestClient; não inclui overhead de rede ou uvicorn. Requer teste de carga externo para validação em produção real.
