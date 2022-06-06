# Importa algumas funções do pacote flask para fazer o desenvolvimento de páginas web com python
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for
import pandas as pd # Maniulação de arquivos     
import numpy as np # operações matematicas com arrays
import sklearn as sk # importa pacate scikit-learn, onde traz algoritmos e recurso de Machine Learnig

from modelo_turbidez import dtr, modelo_v1 # importa variveis da arquivo modelo_turbidez.py
from modelo_solidos_totais import dts, modelo_v2 # importa variveis da arquivo modelo_solidos_totais.py
import os # importa biblioteca para entrada e saída de arquivo

app = Flask(__name__)
#####################################################################################################################
@app.route("/") # rota principal
def index():

    
    #var7= X
    #var8=Y
    #var9=df2
    return render_template('/pagina_principal/principal.html') # renderiza na página o arquivo HTML e passa o nome da rota

#####################################################################################################################
@app.route("/results_tres_marias_turbidez", methods=["GET", "POST"]) # Paginas de resultados
# GET = Acessa infromações no servidor
# POST = Manda informações para o servidor

# Predição Turbidez
def resultado_turbidez():
    
    if request.method == "GET": # Verifica se foi feita a busca de dados no servidor

        return render_template('/resultado_turbidez/tres_marias.html') # renderiza na página o arquivo HTML e passa o nome da rota
    
    else:

        # Acessa nível de cinza passadas para predizer valor de turbidez
        par = str(request.form.get("parametro"))

        # Acessa nível de cinza passadas para predizer valor de turbidez
        valor = str(request.form.get("entrada"))

        def is_number(num): # Função para verificar se string e numero
        
            try:
                #Try to convert the input. 
                float(num)
                
                #If successful, returns true.
                return True # String é Numero
                
            except:
                #Silently ignores any exception.
                pass # ignora excessões
            
            #If this point was reached, the input is not a number and the function
            #will return False.
            return False # String não e numero

        valor_number = is_number(valor) # Verifica se strig é numero
        par_string = is_number(par) # Verifica se string é string

        if par=="" or valor=="" or valor_number==False or par_string==True or (par!="B2" and par!="B3" and par!="B4" and par!="cor_verdadeira_S" and par!="sulfato_S" and par!="turbidez_ZF"): # Condições para fazer predição de turbidez
            
            return render_template('/resultado_turbidez/tres_marias_dados.html') # renderiza na página o arquivo HTML e passa o nome da rota
        #############################################################
        # Predizendo valores com entradas informadas pelo usuario

        # Média dos dados contidos nas colunas dos dados de treinamento (X_treino)
        vec_dados=np.array([[dtr['B2'].mean(), dtr['B3'].mean(), dtr['B4'].mean(), 
                     dtr['cor_verdadeira_S'].mean(), dtr['sulfato_S'].mean(), dtr['turbidez_ZF'].mean()]])
        
        # Cria Dataframe com média dos dados de treinamento (X_treino) 
        dtp=pd.DataFrame(vec_dados, columns=['B2', 'B3', 'B4', 'cor_verdadeira_S', 'sulfato_S', 'turbidez_ZF'])

        if par=="B2" or par=="B3" or par=="B4": # verifica se ira fazer predição com bandas ou parâmetros limnologicos

            # Adiciona valores realciondos algum dos parâmetros preditores (B2, B3, B4)
            dtp[par] = np.array([[float(valor)/10]]) # A divisão por 10 e para fazer a normalização do valor a ser predito (NORMALIZAÇÃO APENAS PARA AS BANDAS)
        else:
            
            # Adiciona valores realciondos algum dos parâmetros preditores (cor_verdadeira_S, sulfato_S, turbidez_ZF)
            dtp[par] = np.array([[float(valor)]]) # Nova variavel preditora

        # Predição de novos valores com dados de entrada do usuario
        pred=modelo_v1.predict(dtp)

        if pred < 0: # Se valor de resultado de predição for menor que zero

            return render_template('/resultado_turbidez/tres_marias_resultado_negativo.html') # renderiza na página o arquivo HTML e passa o nome da rota

        return render_template('/resultado_turbidez/tres_marias.html', variavel=par, variavel1=pred) # renderiza na página o arquivo HTML e passa o nome da rota

#####################################################################################################################
@app.route("/results_tres_marias_solidos_totais", methods=["GET", "POST"]) # Paginas de resultados
# GET = Acessa infromações no servidor
# POST = Manda informações para o servidor

# Predição Sólidos totais
def resultado_solidos_totais():
    
    if request.method == "GET": # Verifica se foi feita a busca de dados no servidor

        return render_template('/resultado_solidos_totais/tres_marias.html') # renderiza na página o arquivo HTML e passa o nome da rota
    
    else:

        # Acessa nível de cinza passadas para predizer valor de turbidez
        par = str(request.form.get("parametro"))

        # Acessa nível de cinza passadas para predizer valor de turbidez
        valor = str(request.form.get("entrada"))

        def is_number(num): # Função para verificar se string e numero
        
            try:
                #Try to convert the input. 
                float(num)
                
                #If successful, returns true.
                return True # String é Numero
                
            except:
                #Silently ignores any exception.
                pass # ignora excessões
            
            #If this point was reached, the input is not a number and the function
            #will return False.
            return False # String não e numero

        valor_number = is_number(valor) # Verifica se strig é numero
        par_string = is_number(par) # Verifica se string é string

        if par=="" or valor=="" or valor_number==False or par_string==True or (par!="B1" and par!="B2" and par!="B4" and par!="B55" and par!="B6" and par!="B7" and par!="sdt_S" and par!="sulfato_S"): # Condições para fazer predição de turbidez
            
            return render_template('/resultado_solidos_totais/tres_marias_dados.html') # renderiza na página o arquivo HTML e passa o nome da rota
        #############################################################
        # Predizendo valores com entradas informadas pelo usuario

        # Média dos dados contidos nas colunas dos dados de treinamento (X_treino)
        vec_dados=np.array([[dts['B1'].mean(), dts['B2'].mean(), dts['B4'].mean(), dts['B55'].mean(), 
                            dts['B6'].mean(), dts['B7'].mean(), dts['sdt_S'].mean(), dts['sulfato_S'].mean()]])
        
        # Cria Dataframe com média dos dados de treinamento (X_treino) 
        dtp=pd.DataFrame(vec_dados, columns=['B1', 'B2', 'B4', 'B55', 'B6', 'B7', 'sdt_S', 'sulfato_S'])

        if par=="B1" or par=="B2" or par=="B4" or par=="B55" or par=="B6" or par=="B7": # verifica se ira fazer predição com bandas ou parâmetros limnologicos

            # Adiciona valores realciondos algum dos parâmetros preditores (B2, B3, B4)
            dtp[par] = np.array([[float(valor)/10]]) # A divisão por 10 e para fazer a normalização do valor a ser predito (NORMALIZAÇÃO APENAS PARA AS BANDAS)
        else:
            
            # Adiciona valores realciondos algum dos parâmetros preditores (cor_verdadeira_S, sulfato_S, turbidez_ZF)
            dtp[par] = np.array([[float(valor)]]) # Nova variavel preditora

        # Predição de novos valores com dados de entrada do usuario
        pred=modelo_v2.predict(dtp)

        if pred < 0: # Se valor de resultado de predição for menor que zero

            return render_template('/resultado_solidos_totais/tres_marias_resultado_negativo.html') # renderiza na página o arquivo HTML e passa o nome da rota

        return render_template('/resultado_solidos_totais/tres_marias.html', variavel=par, variavel1=pred) # renderiza na página o arquivo HTML e passa o nome da rota

#####################################################################################################################
if __name__=='main':
    port = int(os.getenv('PORT'), '5050')
    app.run(host='0.0.0.0', port = port)

if __name__=="__main__": # Verifica se __name__ esta contido neste script
    app.run(debug=True) # Executa este script, definindo endereço do servidor que ira rodar a aplicação e a porta


# COMANDO PARA CRIAR ARQUIVO REQUIREMENTS.TXT
# pip list --format=freeze > requirements.txt

#ACESSAR SITE
# https://plattform.herokuapp.com/

#PALETA DE CORES USADA_Dividir complementar duas vezes
#0117DB (principal fica no meio)
#670BDB
#DB8816
#DBD816
#0B8DDB