##📊 Análise de Rotatividade de Colaboradores
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#📁 Visão Geral
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Este projeto visa analisar e prever a rotatividade de colaboradores em uma organização, utilizando duas abordagens complementares:

Análise Diagnóstica (Power BI): Examina os padrões e características dos colaboradores que já se desligaram, comparando-os com os colaboradores ativos.

Análise Preditiva (Python): Desenvolve um modelo de machine learning para prever quais colaboradores têm maior probabilidade de solicitar desligamento no futuro.



#🧠 Análise Diagnóstica com Power BI
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
A análise diagnóstica foi conduzida no Power BI, permitindo uma visualização interativa dos dados relacionados aos colaboradores. Os principais objetivos desta análise incluem:

Identificar padrões demográficos e profissionais entre os colaboradores que se desligaram.

Comparar métricas de desempenho e engajamento entre colaboradores ativos e desligados.

Detectar possíveis fatores que contribuem para a rotatividade.

O dashboard desenvolvido oferece insights valiosos para a equipe de Recursos Humanos, auxiliando na compreensão dos motivos de desligamento e na formulação de estratégias de retenção.
![image](https://github.com/user-attachments/assets/fdc11bcf-33f0-4e34-b6a5-6e1a6b4280ea)

#🤖 Análise Preditiva com Python
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
A análise preditiva foi realizada utilizando Python, com o objetivo de antecipar quais colaboradores têm maior probabilidade de solicitar desligamento. As etapas principais incluem:

Pré-processamento de Dados:

Limpeza e tratamento de dados ausentes.

Codificação de variáveis categóricas.

Normalização de variáveis numéricas.

Modelagem:

Divisão dos dados em conjuntos de treino e teste.

Treinamento de modelos de classificação (e.g., Random Forest, XGBoost).

Avaliação dos modelos utilizando métricas como acurácia, precisão, recall e F1-score.

Implementação:

Seleção do modelo com melhor desempenho.

Geração de previsões para o conjunto de dados atual.

Integração dos resultados com o Power BI para visualização.


#🛠️ Tecnologias Utilizadas
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Power BI: Visualização e análise de dados.

Python: Análise preditiva e manipulação de dados.

Bibliotecas: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn.
