# Análise Comparativa de Métodos de Validação Cruzada no Desempenho do Classificador SVM para Predição de Doenças Cardíacas

## 1. 🎯 Objetivo do Projeto

Este repositório documenta uma análise sobre a aplicação do modelo de aprendizado de máquina **Máquina de Vetores de Suporte (SVM)** para a predição de doenças cardíacas. O foco principal do estudo é uma análise comparativa rigorosa de como diferentes estratégias de validação cruzada (CV) — especificamente **KFold** e **StratifiedShuffleSplit** — impactam a performance estimada e a seleção de hiperparâmetros do classificador.

A escolha do método de validação é fundamental na área biomédica para estimar a capacidade de generalização do modelo, prevenir o superajuste (overfitting) e garantir a confiabilidade dos resultados.

## 2. 💾 Dataset

O estudo utilizou o dataset "Heart Disease (Cleveland)", obtido de um repositório público. Este conjunto de dados contém 303 amostras (pacientes) e 13 atributos clínicos (features), como idade, sexo, tipo de dor no peito (`cp`) e número de vasos principais (`ca`).

O atributo alvo (`target`) é uma classificação binária:
* **0:** Saudável (sem doença cardíaca)
* **1:** Doente (presença de doença cardíaca)

## 3. 🛠️ Metodologia e Pipeline do Modelo

Para garantir a reprodutibilidade e evitar o vazamento de dados (*data leakage*), todo o processo de pré-processamento e modelagem foi encapsulado em um `Pipeline` do Scikit-learn.

O pipeline de classificação consiste nas seguintes etapas:

1.  **Pré-processamento (`StandardScaler`):** Padronização dos atributos numéricos (Z-score) para normalizar a escala das *features* antes de alimentar o modelo SVM.
2.  **Classificação (`SVC`):** Utilização do Classificador de Vetores de Suporte (Support Vector Classifier) como o estimador principal.
3.  **Otimização (`GridSearchCV`):** O `GridSearchCV` foi empregado para realizar uma busca exaustiva (grid search) e encontrar a combinação ótima de hiperparâmetros (`C`, `kernel` e `gamma`) para o SVC.

## 4. 📊 Análise Central: KFold vs. StratifiedShuffleSplit

O `GridSearchCV` foi executado duas vezes, aplicando duas estratégias de CV distintas para avaliar o pipeline:

* **KFold (KFold-10):** Uma CV padrão com 10 partições (folds) sequenciais.
* **StratifiedShuffleSplit (SSS):** Uma CV com 10 divisões, utilizando 25% dos dados para teste em cada divisão. Crucialmente, o SSS garante que a proporção das classes (saudável vs. doente) seja mantida em cada conjunto de treino e teste (estratificação).

## 5. 📈 Resultados e Visualizações

A análise comparativa revelou nuances importantes no desempenho de cada estratégia, conforme detalhado no notebook `SVC_Doencas_Cardiacas.ipynb`.

### Tabela Comparativa dos Melhores Modelos

A tabela abaixo resume os resultados ótimos encontrados pelo `GridSearchCV` para cada estratégia de CV:

| Estratégia CV | Melhor Acurácia Média | Desvio Padrão (Estabilidade) | Melhores Parâmetros Encontrados |
| :--- | :--- | :--- | :--- |
| KFold-10 | 84.13% | 0.0509 | `{'svc__C': 10, 'svc__kernel': 'rbf', 'svc__gamma': 0.001}` |
| StratifiedShuffleSplit | **84.47%** | **0.0226** | `{'svc__C': 0.1, 'svc__kernel': 'rbf', 'svc__gamma': 'scale'}` |

*(Resultados extraídos da Tabela Resumo, Bloco 7 do notebook)*

### Principais Visualizações e Insights

* **Estabilidade do Modelo (Gráficos de Linha):** O SSS (linha vermelha) demonstrou uma acurácia média mais estável (menor variância) em todas as combinações de hiperparâmetros testadas, em comparação com a maior oscilação do KFold (linha roxa).

* **Visualização das Divisões de CV:** A análise visual das partições demonstra como o KFold divide os dados em blocos sequenciais, enquanto o SSS amostra aleatoriamente (e de forma estratificada) o conjunto de dados em cada iteração, fornecendo uma estimativa de generalização mais robusta.

* **Justificativa do Kernel RBF (Heatmap & PCA):** O gráfico de Análise de Componentes Principais (PCA) mostrou que as classes não são linearmente separáveis. Isso justifica por que ambas as estratégias de CV selecionaram o kernel **'rbf'** (não linear) como a melhor opção. O heatmap de sensibilidade (abaixo) ilustra como a acurácia do kernel RBF varia com `C` e `gamma`.

* **Importância das Features (Permutation Importance):** Como o kernel 'rbf' é não linear, a *Permutation Importance* foi utilizada para avaliar o impacto de cada *feature*. A análise (abaixo) mostrou que o modelo RBF baseou suas decisões principalmente em três atributos: `ca` (nº de vasos principais), `thal` (tipo de defeito) e `cp` (tipo de dor no peito).

## 6. 🏁 Conclusão

Embora ambas as estratégias tenham alcançado uma acurácia média semelhante (KFold 84.13% vs SSS 84.47%), o **StratifiedShuffleSplit (SSS) provou ser a metodologia superior** para este problema.

A principal vantagem do SSS foi sua **estabilidade**: o Desvio Padrão significativamente menor (0.0226) indica que seu desempenho é mais consistente e confiável entre diferentes amostras de dados, em comparação com o KFold (0.0509).

Além disso, a **estratificação** do SSS é uma prática recomendada para dados biomédicos, que frequentemente são desbalanceados, garantindo que o modelo seja avaliado de forma justa em ambas as classes.

## 7. 🚀 Como Executar

O projeto está contido no Jupyter Notebook `SVC_Doencas_Cardiacas.ipynb`.

1.  Clone este repositório.
2.  Certifique-se de ter as bibliotecas listadas no Bloco 1 do notebook instaladas (numpy, pandas, matplotlib, seaborn, scikit-learn).
3.  Execute o notebook em um ambiente Jupyter (como Jupyter Lab, VS Code ou Google Colab).
