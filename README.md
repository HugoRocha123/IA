# NEO — Previsão de Objetos Perigosos Próximos da Terra

Notebooks:
- `notebooks/neo_projeto_completo.ipynb` (Business/Data Understanding + Data Preparation)
- `notebooks/neo_modeling.ipynb` (Modeling)

Dataset: `Datasets/data/raw/neo.csv` (NASA — Near-Earth Objects)

## 1. Business Understanding

**Tipo de tarefa:** Classificação binária supervisionada.

**Entidade das previsões:** Cada NEO (Near-Earth Object) individual — asteroide ou cometa identificado por `id`/`name`.

**Possíveis resultados a prever:** `hazardous = True` (potencialmente perigoso) ou `hazardous = False` (não perigoso).

**Quando são observados os resultados:** O rótulo `hazardous` já vem definido no dataset histórico, atribuído pela NASA/JPL com base em critérios orbitais (distância mínima de interseção orbital — MOID) e físicos (tamanho estimado do objeto). O modelo não calcula o perigo em tempo real; aprende um padrão a partir de exemplos já rotulados e aplica-o a objetos novos.

## 2. Data Understanding

- **Dimensão:** 90.836 registos, 10 colunas.
- **Qualidade:** sem valores em falta, sem duplicados.
- **Variável-alvo (`hazardous`):** desequilibrada — ~90,3% `False` vs. ~9,7% `True`.
- **Colunas sem valor preditivo:** `orbiting_body` e `sentry_object` são constantes; `id` e `name` são identificadores.
- **Redundância detetada:** `est_diameter_min` e `est_diameter_max` têm correlação de 1.0 (perfeita).
- **Outliers:** presentes em `est_diameter_max` (~9%) e `relative_velocity` (~2%), mas correspondem a objetos fisicamente plausíveis, não a erros.

Detalhe completo da análise (histogramas, boxplots, matriz de correlação) no notebook `neo_projeto_completo.ipynb`, secção 2.

## 3. Data Preparation

- Removidas as colunas `id`, `name`, `orbiting_body`, `sentry_object`.
- Removida `est_diameter_min` por redundância; criada `diameter_mean` como feature alternativa.
- `hazardous` convertida de booleano para inteiro (0/1).
- Split treino/teste 80/20, com `stratify` para preservar a proporção de classes.
- Outliers mantidos (não removidos) por serem fisicamente válidos.
- Variáveis numéricas escaladas com `StandardScaler` (ajustado apenas no treino, para evitar data leakage).
- Desequilíbrio de classes documentado, a tratar na fase de Modeling (`class_weight` ou `scale_pos_weight`).

Detalhe completo no notebook `neo_projeto_completo.ipynb`, secção 3.

Dados preparados guardados em: `Datasets/data/raw/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`.

## 4. Modeling

Foram treinados e comparados 3 modelos de classificação, todos com tratamento do desequilíbrio de classes (`class_weight='balanced'` ou `scale_pos_weight`):

| Modelo | Recall (Perigoso) | Precision (Perigoso) | ROC-AUC |
|---|---|---|---|
| Regressão Logística | 93,2% | 30,4% | 0,879 |
| Random Forest | 63,7% | 47,7% | 0,934 |
| **XGBoost** | **96,9%** | 32,6% | 0,924 |

**Validação:** os resultados foram confirmados com 5-fold cross-validation (desvios-padrão < 1%, resultados estáveis e não dependentes do split escolhido).

**Ajuste de limiar:** o Random Forest, apesar do melhor ROC-AUC, tinha o pior recall com o limiar padrão (0.5). Ao baixar o limiar para 0.2, o recall subiu para 92,3% (precision: 36,7%), tornando-se uma alternativa competitiva ao XGBoost.

**Modelo escolhido: XGBoost.** Neste problema, o erro mais grave é o falso negativo (classificar um objeto realmente perigoso como não perigoso), por isso o recall foi a métrica prioritária. O XGBoost tem o melhor recall (95,7% em cross-validation), com uma precision mais baixa (32,6%) considerada um custo aceitável face ao risco de não detetar um objeto genuinamente perigoso.

**Variáveis mais importantes:** `est_diameter_max` foi a mais relevante em ambos os modelos de árvore, mas o XGBoost concentra ~85% da importância nela, enquanto o Random Forest distribui de forma mais equilibrada pelas 5 variáveis.

Detalhe completo (matrizes de confusão, curvas ROC, gráficos de importância) no notebook `notebooks/neo_modeling.ipynb`.

## Estrutura de pastas
