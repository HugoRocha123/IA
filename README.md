# Language Identification — IA III / ML III

Executar **na raiz desta pasta**, por ordem:

1. `01_Sprint1_Business_Data_Understanding.ipynb` — ML Canvas, contrato de dados, EDA
2. `02_Sprint2_Data_Preparation.ipynb` — limpeza, split, features (gera `data/` e `models/prep_*.joblib`)
3. `03_Sprint3_Modeling.ipynb` — modelos sklearn + TensorFlow (gera `models/fitted_models.joblib`)
4. `04_Sprint4_Evaluation_Deployment.ipynb` — avaliação final, multiagente, simulação

`src/` contém os módulos reutilizáveis: `utils.py`, `data_loader.py`, `data_contract.py`, `agent.py`.

Dependências: `pip install kagglehub scikit-learn tensorflow xgboost transformers seaborn joblib requests`
