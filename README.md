# NEO — Previsão de Objetos Perigosos Próximos da Terra

Dataset: `Datasets/neo/raw/neo.csv` (NASA — Near-Earth Objects)
Notebook: `notebooks/neo_projeto_completo.ipynb`

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

Detalhe completo da análise (histogramas, boxplots, matriz de correlação) no notebook, secção 2.

## 3. Data Preparation

- Removidas as colunas `id`, `name`, `orbiting_body`, `sentry_object`.
- Removida `est_diameter_min` por redundância; criada `diameter_mean` como feature alternativa.
- `hazardous` convertida de booleano para inteiro (0/1).
- Split treino/teste 80/20, com `stratify` para preservar a proporção de classes.
- Outliers mantidos (não removidos) por serem fisicamente válidos.
- Variáveis numéricas escaladas com `StandardScaler` (ajustado apenas no treino, para evitar data leakage).
- Desequilíbrio de classes documentado, a tratar na fase de Modeling (`class_weight` ou SMOTE).

Detalhe completo no notebook, secção 3.

## Estrutura de pastas
