### Relatório de Análise do Dataset Wine Utilizando K-Means

### Introdução

O objetivo desta análise foi aplicar o algoritmo K-Means ao dataset Wine para explorar a capacidade de agrupamento do algoritmo com base nas características químicas dos vinhos. A tarefa era identificar padrões nos dados que permitissem agrupar os vinhos em categorias correspondentes às suas cultivares originais, utilizando um método de aprendizado não supervisionado.

### Passos Realizados

### 1. Exploração do Dataset

- **Carregamento e Descrição**: Utilizamos o dataset Wine do scikit-learn, que contém 178 amostras de vinhos com 13 características químicas diferentes e 3 categorias de cultivares (0, 1, 2).
- **Estrutura dos Dados**:
    - Amostras: 178
    - Características: 13 (como álcool, ácido málico, cinzas, etc.)
    - Target: Categoria do vinho (0, 1, 2)

### 2. Pré-processamento dos Dados

- **Normalização**: Aplicamos a normalização usando `StandardScaler` para garantir que todas as características tivessem a mesma escala, com média 0 e desvio padrão 1, facilitando o processo de agrupamento.

```python
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

```

### 3. Determinação do Número de Clusters (k)

- **Elbow Method**: Utilizamos o método do cotovelo para determinar o número ideal de clusters. Plotamos a inércia para diferentes valores de k e identificamos o "cotovelo" no gráfico, que sugeriu 3 clusters como um ponto de inflexão.

```python
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_normalized)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.title('Método do Cotovelo')
plt.show()

```

### 4. Aplicação do K-Means

- **Configuração e Treinamento**: Aplicamos o algoritmo K-Means com k=3 ao dataset normalizado.
- **Armazenamento dos Labels**: Armazenamos os labels dos clusters resultantes para posterior análise.

```python
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_normalized)
df_normalized['cluster'] = kmeans.labels_

```

### 5. Avaliação do Modelo

- **Matriz de Confusão**: Calculamos a matriz de confusão para comparar os clusters formados com as classificações reais dos vinhos.
- **Métricas de Avaliação**: Utilizamos o índice de Rand ajustado e a acurácia para avaliar o desempenho do modelo.

```python
conf_matrix = confusion_matrix(df_normalized['target'], mapped_labels)
adjusted_rand = adjusted_rand_score(df_normalized['target'], labels)
accuracy = accuracy_score(df_normalized['target'], mapped_labels)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Clusters Preditos')
plt.ylabel('Etiquetas Reais')
plt.title('Matriz de Confusão entre Clusters e Etiquetas Reais')
plt.show()

```

### Resultados Obtidos

- **Matriz de Confusão**:

```markdown
   Predito 0 | Predito 1 | Predito 2
-------------------------------------
0 |      0   |      59   |      0
1 |      2   |      3    |      66
2 |     48   |      0    |      0

```

- **Métricas de Avaliação**:
    - Índice de Rand Ajustado: Muito baixo, indicando baixa correspondência entre os clusters formados e as categorias reais.
    - Acurácia: Muito baixa, refletindo a alta taxa de erros de classificação.

### Conclusões

- **Desempenho Insatisfatório**: O K-Means com 3 clusters não conseguiu identificar corretamente as categorias de vinhos com base nas características químicas. A matriz de confusão mostrou que muitos vinhos foram agrupados incorretamente.
- **Análise de Clusters**: A maior parte das amostras da classe 0 foi agrupada na classe 1, e a classe 2 foi agrupada na classe 0, sugerindo uma sobreposição significativa entre as características das classes.
- **Possíveis Melhorias**:
    - **Reavaliar o Número de Clusters**: Testar outros números de clusters (por exemplo, 2 ou 4) para verificar se há uma melhor correspondência.
    - **Técnicas de Clustering Alternativas**: Explorar algoritmos de clustering diferentes, como DBSCAN ou Agglomerative Clustering, que podem lidar melhor com a distribuição dos dados.
    - **Redução de Dimensionalidade**: Utilizar PCA (Análise de Componentes Principais) para reduzir a dimensionalidade dos dados e identificar combinações de características que melhor separam as classes.

Em resumo, embora o K-Means seja uma técnica poderosa de agrupamento, neste caso específico do dataset Wine, ele não foi eficaz em identificar as categorias originais dos vinhos. A análise sugere a necessidade de uma abordagem mais refinada ou de técnicas alternativas para obter melhores resultados.
