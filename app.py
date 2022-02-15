# Importa algumas funções do pacote flask para fazer o desenvolvimento de páginas web com python
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for
import pandas as pd # Maniulação de arquivos     
#import matplotlib as mat # Analise gráfica visual
#import matplotlib.pyplot as plt # PLotagem de graficos
import numpy as np # operações matematicas com arrays
import sklearn as sk # importa pacate scikit-learn, onde traz algoritmos e recurso de Machine Learnig

from machine import dtr, modelo_v1
import os # importa biblioteca para entrada e saída de arquivo

app = Flask(__name__)


@app.route("/") # rota principal
def index():

    
    #var7= X
    #var8=Y
    #var9=df2
    return render_template('principal.html') # renderiza na página o arquivo HTML e passa o nome da rota

@app.route("/results_tres_marias", methods=["GET", "POST"]) # Paginas de resultados
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


#if __name__=="__main__": # Verifica se __name__ esta contido neste script
    #app.run(debug=True) # Executa este script, definindo endereço do servidor que ira rodar a aplicação e a porta
if __name__=='main':
    port = int(os.getenv('PORT'), '5000')
    app.run(host='0.0.0.0', port = port)


# COMANDO PARA CRIAR ARQUIVO REQUIREMENTS.TXT
# pip list --format=freeze > requirements.txt

#ACESSAE SITE
# https://plattform.herokuapp.com/