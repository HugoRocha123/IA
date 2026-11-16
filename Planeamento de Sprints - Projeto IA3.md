# Planeamento de Sprints — Projeto de IA3 (Previsão de NEOs Perigosos)

Este documento explica como organizei o meu trabalho nesta parte do projeto de **Inteligência Artificial 3 (IA3)**, que desenvolvi de forma individual. Dividi o processo em **4 Sprints**, juntando uma forma de trabalho ágil às etapas da metodologia **CRISP-DM**.

## Resumo das Sprints

| Sprint | Fase CRISP-DM | O que fiz | Principal Resultado |
| :---- | :---- | :---- | :---- |
| **Sprint 1** | *Business & Data Understanding* | Perceber o problema e explorar os dados da NASA | Descobri que apenas \~9,7% dos 90.836 registos são perigosos |
| **Sprint 2** | *Data Preparation* | Limpar os dados e escolher as variáveis mais úteis | Dados prontos e divididos corretamente entre treino e teste |
| **Sprint 3** | *Modeling* | Treinar e testar três modelos de previsão diferentes | Regressão Logística, Random Forest e XGBoost validados |
| **Sprint 4** | *Evaluation & Deployment* | Comparar resultados, afinar detalhes e fechar o projeto | Escolha do **XGBoost** (96,9% de *Recall*) e entrega final |

## O que foi feito em cada Sprint

### Sprint 1: Perceber o Problema e Explorar os Dados

* **Fase CRISP-DM:** *Business Understanding* & *Data Understanding*  
* **Objetivo:** Entender o tema (objetos que passam perto da Terra), definir o que queria prever e analisar os dados da NASA.  
* **O que fiz:**  
  * Defini o objetivo: prever se um objeto espacial é perigoso ou não (`hazardous`).  
  * Decidi dar prioridade máxima ao **Recall**, porque deixar escapar um asteroide perigoso (falso negativo) é muito mais grave do que dar um falso alarme.  
  * Analisei os 90.836 registos e reparei que os dados estão muito desequilibrados (cerca de 90,3% não são perigosos e só 9,7% são perigosos).  
  * Fui ver que colunas faziam mais diferença, destacando-se logo o tamanho estimado (`est_diameter_max`) e a velocidade (`relative_velocity`).  
* **Resultado:** Um primeiro notebook com a análise dos dados e as regras do jogo bem definidas.

### Sprint 2: Limpar e Preparar os Dados

* **Fase CRISP-DM:** *Data Preparation*  
* **Objetivo:** Deixar os dados limpos e prontos para os algoritmos conseguirem aprender sem erros.  
* **O que fiz:**  
  * Limpei valores em falta, registos repetidos e tirei colunas que não ajudavam na previsão (como IDs ou nomes).  
  * Escolhi as 5 variáveis principais que tinham mais peso para o problema.  
  * Ajustei a escala dos valores para que nenhum algoritmo ficasse baralhado com números muito grandes ou muito pequenos.  
  * Dividi os dados em treino e teste, garantindo que a proporção de objetos perigosos se mantinha igual nos dois lados.  
* **Resultado:** Dados limpos e prontos a entrar nos modelos.

### Sprint 3: Treinar e Validar os Modelos

* **Fase CRISP-DM:** *Modeling*  
* **Objetivo:** Testar diferentes algoritmos e compensar o facto de haver poucos exemplos de objetos perigosos.  
* **O que fiz:**  
  * Comecei com um modelo mais simples como base: a **Regressão Logística** (ajustando os pesos das classes).  
  * Passei para modelos mais avançados baseados em árvores: **Random Forest** e **XGBoost**, também configurados para dar mais atenção aos casos perigosos.  
  * Usei validação cruzada (*5-fold CV*) para ter a certeza de que os resultados eram fiáveis e não fruto do acaso.  
  * Guardei as métricas principais (*Recall*, *Precision*, *F1-Score* e *ROC-AUC*) para comparar tudo mais à frente.  
* **Resultado:** Os três modelos treinados e testados no notebook `neo_modeling.ipynb`.

### Sprint 4: Comparar Resultados, Afinar e Fechar o Projeto

* **Fase CRISP-DM:** *Evaluation* & *Deployment*  
* **Objetivo:** Escolher o melhor modelo para o meu objetivo, perceber como ele decide e organizar toda a documentação.  
* **O que fiz:**  
  * Comparei os resultados finais nos dados de teste:  
    * **XGBoost:** 96,9% *Recall* \| 32,6% *Precision* \| 0,924 *ROC-AUC*  
    * **Regressão Logística:** 93,2% *Recall* \| 30,4% *Precision* \| 0,879 *ROC-AUC*  
    * **Random Forest:** 63,7% *Recall* \| 47,7% *Precision* \| 0,934 *ROC-AUC*  
  * Testei baixar o limiar de decisão (*threshold tuning*) do Random Forest de `0.5` para `0.2`, o que fez o *Recall* subir para 92,3%.  
  * Vi quais as variáveis que mais pesavam nas decisões e confirmei que o diâmetro máximo (`est_diameter_max`) é o fator mais decisivo.  
  * Escolhi o **XGBoost** como modelo final por ser o mais seguro a detetar ameaças e escrevi o `README_2.md` para explicar o trabalho todo.  
* **Resultado:** Projeto concluído, modelo escolhido e documentação pronta a entregar.