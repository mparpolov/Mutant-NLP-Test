# Desafio Mutant - Analista de NLP Jr

### Algoritmo Escolhido
- KNN: K Nearest Neighbors

### Métricas
As métricas abaixo foram geradas utilizando 70% do Dataset para trienamento e 30% para testes:

- Número ótimo de K: 5
- Porcentagem de Acertos: 96,67%
### Tabelas de Classificações:
#### Por Classe:
| - |  precision  |  recall | f1-score |  support |
| ------ | ------ | ------ | ----- | ----- |
| Iris-Setosa | 1.00 | 1.00 | 1.00 | 7 |
| Iris-Virginica | 0.92 | 1.00 | 0.96 | 12 |
| Iris-Versicolor | 1.00 | 0.91 | 0.95 | 11 |
#### Geral:
| - |  precision  |  recall | f1-score |  support |
| ------ | ------ | ------ | ----- | ----- |
| accuracy | - | - | 0.97 | 30 |
| macro avg | 0.97 | 0.97 | 0.97 | 30 |
| weighted avg | 0.97 | 0.97 |0.97 | 30 |

Conforme os gráficos abaixo, as previsões realizadas pelo modelo tiveram alto indícide de acertos:
![Iris Datasets](https://github.com/mauUsatai/Mutant-NLP-Test/blob/master/iris_classification.png)

### Seu modelo obteve um bom resultado de classificação?
Sim, o modelo pode classificar dados não encontrados antes com 96,67% de precisão.

### Como você fez para avaliar?
Os dados necessários para a avaliação da precisão do modelo podem ser encontrados nas tabelas acima. De acordo com a tabela por classes, o modelo obteve ótimos resultados apresentando 100% de acerto (verdadeiros positivos) nas classes Setosa e Versicolor. Obtendo também 92% de verdadeiros positivos na classe Virginica. Também de acordo com as tabelas, o modelo possui baixo índice de falsos negativos (recall).
