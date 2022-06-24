import pandas as pd # Maniulação de arquivos     
import numpy as np # operações matematicas com arrays
import sklearn as sk # importa pacate scikit-learn, onde traz algoritmos e recurso de Machine Learnig

# Carregando arquivo csv (arquivos separados por vírgulas pelo Excel) do banco de dados
# usecols, seleciona colunas que possivelmente serão usadas no treinamento do algoritmo
df2 = pd.read_csv("Dados_ajustados_completo_uso.csv", usecols = ['B1', 'B2', 'B4', 'B55', 'B6', 'B7', 'sdt_S', 'sulfato_S', 'solidos_totais_S'])

# Divide valores dos dados das colunas referentes as informações espectrais de satélite por 10, para que essas variaveis não tenha um peso maior na etapa de treinamento do modelo de aprendizagem

df2['B1']=df2['B1']/10 # Banda 1
df2['B2']=df2['B2']/10 # Banda 2
df2['B4']=df2['B4']/10 # Banda 4
df2['B55']=df2['B55']/10 # Banda 5
df2['B6']=df2['B6']/10 # Banda 6
df2['B7']=df2['B7']/10 # Banda 7


from sklearn.model_selection import train_test_split # Importa função do scikit learn para fazer a separação dos dados em treinamento (70%) e teste (30%)

# Seleção de variáveis preditoras (variaveis indepedentes), que são as colunas do dataset que contem as informações das bandas das imagens e dados 
# limnologicos de outros parâmetros de qualidade de água, para fazer a predição do parâmetro de qualidade da água escolhido
atributos = ['B1', 'B2', 'B4', 'B55', 'B6', 'B7', 'sdt_S', 'sulfato_S'] # Colunas do dataset

# Variável a ser prevista (variavel depedente), e a coluna no dataset que vai informar a quantidade de turbidez que contém na água
atrib_prev = ['solidos_totais_S'] # Coluna do datasett

# Criando objetos (formato numpy - array)
X = df2[atributos].values # Variaveis indepedente ou preditoras
Y = df2[atrib_prev].values # Variavel depdente ou target

# Definindo a taxa de split (separação da quantidade de dados para teste) que neste caso é 30%
split_test_size = 0.30 # 30% do dataset vai ser usado como teste, logo automaticamnete o algoritmo vai entender que os 70% são usados para treinamento

# Criando dados de treino e de teste
# Faz separação dos dados de treinamento (70%) e teste (30%)
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = split_test_size, random_state = 42)


from sklearn.impute import SimpleImputer # Importa função do scikit learn para substituir os valores igual a zero pela média dos outros valores da coluna

# Criando objeto
preenche_0 = SimpleImputer(missing_values = np.nan, strategy = "mean") # Onde há valores igual a zero substitui pela média dos outros valores da coluna

# Substituindo os valores iguais a zero, pela média dos dados na coluna do variavel preditora e variavel target
X_treino = preenche_0.fit_transform(X_treino) # Dados de entrada de treinamento
Y_treino = preenche_0.fit_transform(Y_treino) # Dados de saída de treinamento
X_teste = preenche_0.fit_transform(X_teste) # Dados de entrada de teste
Y_teste = preenche_0.fit_transform(Y_teste) # Dados de saída de teste

# Converte dados de entrada de treinamento em formato numpy (array) para dataframe
dts=pd.DataFrame(X_treino, columns=['B1', 'B2', 'B4', 'B55', 'B6', 'B7', 'sdt_S', 'sulfato_S'])

# Converte dados de saída de treinamento em formato numpy (array) para dataframe
ds=pd.DataFrame(Y_treino, columns=['solidos_totais_S'])

from sklearn.linear_model import LassoCV # Retressãoi linear usando validação cruzada

# Criando objeto de regressão de LASSOCV
modelo_v2=LassoCV(cv=5, random_state=0)

# Treinando o modelo de aprendizagem
modelo_v2.fit(X_treino, Y_treino.ravel())
