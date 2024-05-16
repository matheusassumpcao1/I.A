from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, adjusted_rand_score, homogeneity_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, adjusted_rand_score, accuracy_score

def distribuicao_dados(df):
    """
    Exibe a distribuição das colunas numéricas de um DataFrame usando histogramas.
    
    Parâmetros:
    df (pd.DataFrame): O DataFrame cujas distribuições de dados serão exibidas.
    """
    # Verifica se o DataFrame é válido
    if not isinstance(df, pd.DataFrame):
        raise ValueError("O argumento fornecido não é um DataFrame válido.")
    
    # Seleciona apenas colunas numéricas
    num_cols = df.select_dtypes(include='number').columns
    
    # Número de colunas numéricas
    num_features = len(num_cols)
    
    # Configura a figura para os histogramas
    rows = (num_features + 2) // 3
    fig, axes = plt.subplots(nrows=rows, ncols=3, figsize=(20, 5 * rows), constrained_layout=True)
    axes = axes.flatten() # Aplanar a matriz de eixos
    
    # Gera histogramas para cada coluna numérica
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], bins=20, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribuição da característica: {col}', fontsize=12)
    
    # Remove eixos vazios se o número de características não for múltiplo de 3
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    #Troca o nome da janela que sera aberta
    fig.canvas.manager.set_window_title('Distribuição de Dados')

    plt.show()

def num_amostras(df):
    print(f"Número de amostras: {df.shape[0]}")

def num_caracteristicas(df):
    print(f"Número de caracteristicas: {df.shape[1]}")

def normalizacao(df):
    # Cria um novo DataFrame X contendo todas as colunas de df exceto a coluna target. X contém apenas as características dos vinhos.
    X = df.drop(columns=['target'])
    
    # Cria uma série ( estrutura de dados unidimensional que contém uma sequência de valores com um índice associado a cada valor)
    # contendo apenas a coluna target de df. Y guarda o que foi retirado em dr.drop.
    y = df['target']

    # Normalizar os dados (Média 0 e Desvio Padrão 1)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X) # A combinacao de fit+transform resulta nos dados normalizados.

    #Cria de um novo dataframe, e coloca neles os dados de x_normalized, e da os nomes das colunas originais, assim como adiciona de volta a coluna de rotulos.
    df_normalized = pd.DataFrame(X_normalized, columns=wine.feature_names)
    #A coluna "target" serve para rotular os tipos de vinhos de acordo com as caracteristicas da observacao.
    df_normalized['target'] = y 
    return df_normalized

def clusters(df):
    inertia = []
    for k in range(1, 11):
        # Instanciar e treinar o modelo KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        # Adicionar a inércia à lista
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertia, marker='o')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo para Determinação de k')
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.show()

def k_means(df):
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(df)
    labels = kmeans.labels_
    df['cluster'] = labels
    return labels

def map_clusters_to_labels(clusters, labels):
    # Cria uma matriz de confusão entre clusters e labels
    conf_matrix = confusion_matrix(labels, clusters)
    # Encontra a melhor correspondência entre clusters e labels
    mapping = {}
    for cluster in range(3):
        label = conf_matrix[cluster].argmax()
        mapping[cluster] = label
    return [mapping[cluster] for cluster in clusters]

def calculo_metricas(df, mapeamento):
    # Calcular métricas de avaliação
    conf_matrix = confusion_matrix(df['target'], mapeamento)
    adjusted_rand = adjusted_rand_score(df['target'], labels)
    accuracy = accuracy_score(df['target'], mapeamento)

    print("Matriz de Confusão:")
    print(conf_matrix)
    print(f"Índice de Rand Ajustado: {adjusted_rand:.4f}")
    print(f"Acurácia: {accuracy:.4f}")
    return conf_matrix

def visual_matriz_confusao(conf_matrix):
    # Visualização da Matriz de Confusão
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Clusters Preditos')
    plt.ylabel('Etiquetas Reais')
    plt.title('Matriz de Confusão entre Clusters e Etiquetas Reais')
    plt.show()







#Carregar o dataset Wine e explorar suas características básicas (número de amostras, características, distribuição dos dados, etc.).
wine = load_wine()
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
df['target'] = wine.target
print(df)

num_amostras(df)
num_caracteristicas(df)
distribuicao_dados(df)
print(normalizacao(df).head())
clusters(df) #K=3
df_normalizado = normalizacao(df)
labels = k_means(df_normalizado)
print(df_normalizado)
mapeamento = map_clusters_to_labels(labels, df_normalizado['target'])
conf_matrix = calculo_metricas(df_normalizado, mapeamento)
visual_matriz_confusao(conf_matrix)


