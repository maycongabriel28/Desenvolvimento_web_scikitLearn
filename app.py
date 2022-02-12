# Importa algumas funções do pacote flask para fazer o desenvolvimento de páginas web com python
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for
import pandas as pd # Maniulação de arquivos     
import matplotlib as mat # Analise gráfica visual
import matplotlib.pyplot as plt # PLotagem de graficos
import numpy as np # operações matematicas com arrays
import sklearn as sk # importa pacate scikit-learn, onde traz algoritmos e recurso de Machine Learnig

app = Flask(__name__)

#import os # importa biblioteca para entrada e saída de arquivo

if __name__=='main':
    #port = int(os.getenv('PORT'), '5000')
    app.run(host='0.0.0.0', port = '5000')

# Carregando arquivo csv (arquivos separados por vírgulas pelo Excel) do banco de dados
# usecols, seleciona colunas que possivelmente serão usadas no treinamento do algoritmo
df2 = pd.read_csv("Dados_ajustados_uso.csv", usecols = ['B2', 'B3', 'B4', 'B55', 'cor_verdadeira_S', 'sulfato_S', 'turbidez_S', 'turbidez_ZF'])

from sklearn.model_selection import train_test_split # Importa função do scikit learn para fazer a separação dos dados em treinamento (70%) e teste (30%)

# Seleção de variáveis preditoras (variaveis indepedentes), que são as colunas do dataset que contem as informações das bandas das imagens e dados 
# limnologicos de outros parâmetros de qualidade de água, para fazer a predição do parâmetro de qualidade da água escolhido
atributos = ['B2', 'B3', 'B4', 'B55', 'cor_verdadeira_S', 'sulfato_S', 'turbidez_ZF'] # Colunas do dataset

# Variável a ser prevista (variavel depedente), e a coluna no dataset que vai informar a quantidade de turbidez que contém na água
atrib_prev = ['turbidez_S'] # Coluna do dataset

# Criando objetos (formato numpy - array)
X = df2[atributos].values # Variaveis indepedente ou preditoras
Y = df2[atrib_prev].values # Variavel depdente ou target

# Definindo a taxa de split (separação da quantidade de dados para teste) que neste caso é 30%
split_test_size = 0.30 # 30% do dataset vai ser usado como teste, logo automaticamnete o algoritmo vai entender que os 70% são usados para treinamento

# Criando dados de treino e de teste
# Faz separação dos dados de treinamento (70%) e teste (30%)
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = split_test_size, random_state = 42)

# Verificando se existem valores nulos (NaN) em todo dataset (df1)
df2.isnull().values.any()

from sklearn.impute import SimpleImputer # Importa função do scikit learn para substituir os valores igual a zero pela média dos outros valores da coluna

# Criando objeto
preenche_0 = SimpleImputer(missing_values = 0, strategy = "mean") # Onde há valores igual a zero substitui pela média dos outros valores da coluna

# Substituindo os valores iguais a zero, pela média dos dados na coluna do variavel preditora e variavel target
X_treino = preenche_0.fit_transform(X_treino) # Dados de entrada de treinamento
Y_treino = preenche_0.fit_transform(Y_treino) # Dados de saída de treinamento
X_teste = preenche_0.fit_transform(X_teste) # Dados de entrada de teste
Y_teste = preenche_0.fit_transform(Y_teste) # Dados de saída de teste

# Converte dados de entrada de treinamento em formato numpy (array) para dataframe
dtr=pd.DataFrame(X_treino, columns=['B2', 'B3', 'B4', 'B55', 'cor_verdadeira_S', 'sulfato_S', 'turbidez_ZF'])

# Converte dados de saídaa de treinamento em formato numpy (array) para dataframe
dt=pd.DataFrame(Y_treino, columns=['turbidez_S'])

from sklearn.linear_model import LinearRegression # Importando modulo machine learning de regressão linear

# Criando objeto de regressão linear
modelo_v1 = LinearRegression()

# Treinando o modelo de aprendizagem
modelo_v1.fit(X_treino, Y_treino)

# Importação da função explained_variance_score
from sklearn.metrics import explained_variance_score

# Calculo da métrica estatisca (quanto mais proximo de 1.00 melhor, valores mais baixos são piores)
met1= explained_variance_score(modelo_v1.predict(X_teste), Y_teste)

# Importação da função mean_absolute_error
from sklearn.metrics import mean_absolute_error

# Calcula o erro médio entre duas medidas
met2 = mean_absolute_error(modelo_v1.predict(X_teste), Y_teste)

# Importação da função mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_percentage_error

# Calcula o erro percentual absoluto médio entre duas medidas
met3 = mean_absolute_percentage_error(modelo_v1.predict(X_teste), Y_teste)

# Importação da função r2_score
from sklearn.metrics import r2_score

# Calcula o coeficiente de determninação (R^2), quanto mais proximo de 1 ficar o resultado melhor
met4 = r2_score(modelo_v1.predict(X_teste), Y_teste)


@app.route("/") # rota principal
def index():

    
    var7= X
    var8=Y
    var9=df2
    return render_template('principal.html') # renderiza na página o arquivo HTML e passa o nome da rota

@app.route("/results", methods=["GET", "POST"]) # Paginas de resultados
# GET = Acessa infromações no servidor
# POST = Manda informações para o servidor
def resultado():
    
    if request.method == "GET": # Verifica se foi feita a busca de dados no servidor


        return render_template('tres_marias.html') # renderiza na página o arquivo HTML e passa o nome da rota
    
    else:

        # Acessa nível de cinza passadas para predizer valor de turbidez
        par = str(request.form.get("parametro"))

        # Acessa nível de cinza passadas para predizer valor de turbidez
        valor = int(request.form.get("entrada"))

        #############################################################
        # Predizendo valores com entradas informadas pelo usuario

        # Média dos dados contidos nas colunas dos dados de treinamento (X_treino)
        vec_dados=np.array([[dtr['B2'].mean(), dtr['B3'].mean(), dtr['B4'].mean(), dtr['B55'].mean(), 
                     dtr['cor_verdadeira_S'].mean(), dtr['sulfato_S'].mean(), dtr['turbidez_ZF'].mean()]])
        
        # Cria Dataframe com média dos dados de treinamento (X_treino) 
        dtp=pd.DataFrame(vec_dados, columns=['B2', 'B3', 'B4', 'B55', 'cor_verdadeira_S', 'sulfato_S', 'turbidez_ZF'])


        # Adiciona um novo valor de 'nível de cinza' na banda 4 do landsat 8 para ser usada na predição de turbidez_S
        dtp[par] = np.array([[valor]])

        # Predição de novos valores com dados de entrada do usuario
        pred=modelo_v1.predict(dtp)

        return render_template('tres_marias.html', variavel=par, variavel1=pred) # renderiza na página o arquivo HTML e passa o nome da rota


if __name__=="__main__": # Verifica se __name__ esta contido neste script
    app.run(debug=True) # Executa este script, definindo endereço do servidor que ira rodar a aplicação e a porta