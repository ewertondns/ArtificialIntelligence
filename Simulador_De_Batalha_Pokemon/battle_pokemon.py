#Primeiro importamos as bibliotecas necessárias
import pandas as pandasObj
import numpy as numpyObj
import math

#########################################################
#Aqui nós vamos importar classes da biblioteca Sklearn###
#########################################################
from sklearn import preprocessing, svm 					#
from sklearn.model_selection import train_test_split	#
from sklearn.linear_model import LinearRegression		#			
from sklearn.neighbors import KNeighborsRegressor		#
from sklearn.neighbors import KNeighborsClassifier		#
from sklearn.tree import tree							#
from sklearn.neural_network import MLPRegressor 		#
from sklearn.neural_network import MLPClassifier		#							
from sklearn.svm import LinearSVR						#
from sklearn.svm import LinearSVC						#
from sklearn.ensemble import RandomForestRegressor		#
from sklearn.ensemble import RandomForestClassifier		#
from collections import OrderedDict						#
from PIL import Image									#
from PIL import ImageTk                                 #
from random import randint 								#
import time 											#
import tkinter 											#
import sys                                              #
from tkinter import Tk 									#
from tkinter import Label								#
from tkinter import PhotoImage							#
#########################################################

#Método para tratar imagens e exibir em fullscreen
def showPIL(pilImage):
    root = tkinter.Tk()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.overrideredirect(1)
    root.geometry("%dx%d+0+0" % (w, h))
    root.focus_set()    
    root.bind("<Escape>", lambda e: (e.widget.withdraw(), e.widget.quit()))
    root.after(4000, root.destroy)
    canvas = tkinter.Canvas(root,width=w,height=h)
    canvas.pack()
    canvas.configure(background='black')
    imgWidth, imgHeight = pilImage.size
    if imgWidth > w or imgHeight > h:
        ratio = min(w/imgWidth, h/imgHeight)
        imgWidth = int(imgWidth*ratio)
        imgHeight = int(imgHeight*ratio)
        pilImage = pilImage.resize((imgWidth,imgHeight), Image.ANTIALIAS)
    image = ImageTk.PhotoImage(pilImage)
    imagesprite = canvas.create_image(w/2,h/2,image=image)
    root.mainloop()

#----------------------------------------------Ínicio------------------------------------------------------#
#A funcao pega todas as colunas que foram passadas no dataframe e discretiza os seus valores
#retornando um dataframe com os valores discretizados
def discretiza(colunas):
	#Lembrando que ao recebermos um dicionário no for, ele vai printando suas palavras chave, então basta que
	#a cada palavra chave no for, nós percorramos toda a linha
	novasColunas = {}
	for instancia in colunas:
		palavras = []  #Representará cada um dos nossos rótulos
		valores = {}   #Representará cada um dos valores númericos referentes a esses rótulos
		valor = 0      #O número subirá a partir disso
		rotulados = [] #Representacao de cada elemento classificado, com um rótulo numérico

		for elemento in colunas[instancia]:
			'''
			Vou ver em uma lista se eu já não me deparei com esse
			mesmo elemento anteriomente nessa coluna
			'''
			existe = False

			for x in range(len(palavras)):
				if(len(palavras) == 0):
					#Se não existia nenhum elemento anteriormente, obviamente não havia nada antes para estar igual
					existe = True
					break
				#Se eu encontrar um único elemento igual, eu retiro ela da minha lista
				if(palavras[x] == elemento):
					existe = True
					palavras.remove(elemento)
					#print("Dropei", elemento, end = "")
					break
			palavras.append(elemento)

			#Bom, se o elemento não existir na minha lista, eu o adiciono no meu dicionário de valores
			if(existe == False):
				#print("Elemento:", elemento, "Valor:", valor)
				valores[elemento] = valor
				valor +=1
			#Para cada instancia nessa coluna vamos pegar o valor correspondente do elemento, 
			#em nosso dicionario de valores
			rotulados.append(valores[elemento])
		#Agora seguindo o formato do pandas, eu deixo a palavra chave daquela coluna, para a lista de rotulados
		novasColunas[instancia] = rotulados
	#Retornamos o dataframe convertido
	return pandasObj.DataFrame.from_dict(novasColunas) #Construir um dataframe a partir de um dicionario

def preprocessDates(dataframe):
	#monstros = monstros.drop(['Name', 'ID', 'Sub-Type', 'Attack', 'Full Attack', 'Special Attacks', 'Special Qualities', 'Skills', 'Feats', 'Organization', 'Advancement', 'Dice Type', 'Life Bonus', 'Dex', 'Initiative', 'Base Attack', 'Speed', 'Grapple', 'Space|Reach', 'Treasure'],
	#axis = 1)#Essas colunas eram todas inúteis para a nossa discretização
	dados_reais1 =  dataframe.drop(['#', 'Name', 'Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary'], axis = 1)
	dados_reais2 =  dataframe.drop(['Type 1', 'Type 2', 'Legendary'], axis = 1)

	#print(dados_nreais.head())
	dados_naonumericos1 = dataframe[['Name', 'Type 1', 'Type 2']]
	dados_naonumericos2 = dataframe[['Legendary']]
	#print(dados_naonumericos.head())

	dados_ndiscretos1 = discretiza(dados_naonumericos1)
	dados_ndiscretos2 = discretiza(dados_naonumericos2)
	#print(dados_ndiscretos.head())

	#No caso axis é a coordanada que vamos ir concatenando, se fosse linhas era 0, mas como é colunas ele é 1
	#sorted = false, significa que é para ele colocar na sequência em que isso foi concatenado
	dados_em_processamento = pandasObj.concat([dados_reais1, dados_ndiscretos1], axis=1)
	# print(dados_em_processamento)
	dados_semi_processados = pandasObj.concat([dados_em_processamento, dados_reais2], axis=1)
	# print(dados_semi_processados)
	dados_processados = pandasObj.concat([dados_semi_processados, dados_ndiscretos2], axis=1)
	#print(dados_normalizados.head())

	#print(dados_processados)
	return dados_processados
#----------------------------------------------Fim------------------------------------------------------#

#Aqui nós vamos passar o nosso dataset para um objeto do tipo DataFrame
df1 = pandasObj.read_csv('combats.csv')
df2 = pandasObj.read_csv('pokemon.csv')

# print(df2.tail())

#Aqui estamos usando a função de pré-processamento
pokemon_name = numpyObj.array(df2['Name'])

df2 = preprocessDates(df2)

#Agora vamos passar apenas o elementos que queremos fazer a nossa IA treinar
x1_train = df2.drop(['#', 'Name', 'Type 2',  'Generation'], axis = 1)
y1_train = df2.drop(['Name', 'Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary'], axis = 1)
x1_test = df2.drop(['#', 'Name', 'Type 2',  'Generation'], axis = 1)
y1_test = df2.drop(['Name', 'Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary'], axis = 1)

x2_train = df1.drop(['Winner'], 1)
y2_train = df1.drop(['First_pokemon', 'Second_pokemon'], 1)
x2_test = df1.drop(['Winner'], 1)
y2_test = df1.drop(['First_pokemon', 'Second_pokemon'], 1)
# print(x1_train.tail())
# print(y1_train.tail())
#---------------------------------------------Split--------------------------------------------------#
'''
O Split aqui vai funcionar como o cross validation nessa linha
test_size é igual a quantidade de testes que iremos fazer, por isso temos 0.2 que é referente a 5 nesse caso
logo o cross-validation vai realizar 5 testes, se eu colocar 0.1 ele testaria 10 vezes, como o padrão cross-validation
'''
# x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.30, random_state = 2)
# x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state = 2)
#----------------------------------------------------------------------------------------------------#

#------------------------------------------Decision Tree---------------------------------------------#

#Aqui nós iniciamos o treino para a nossa Árvore de Decisão
tree1 = tree.DecisionTreeClassifier(random_state = 2)#Aqui nós colocamos os parâmetros que nós queremos mudar
tree1 = tree1.fit(x1_train, y1_train)	#Criamos nosso objeto para o treino
accuracyTree1 = tree1.score(x1_test, y1_test)#Aqui nós 

print('Acurácia da Árvore de decisão Classifier: ',accuracyTree1)

#Aqui nós iniciamos o treino para a nossa Árvore de Decisão
tree2 = tree.DecisionTreeRegressor(random_state = 2)#Aqui nós colocamos os parâmetros que nós queremos mudar
tree2 = tree2.fit(x2_train, y2_train)	#Criamos nosso objeto para o treino
accuracyTree2 = tree2.score(x2_test, y2_test)#Aqui nós 

# print('Acurácia da Árvore de decisão(Combat): ',accuracyTree2)

#-----------------------------------------------------------------------------------------------------#


# #----------------------------------------------KNN----------------------------------------------------#

# knn = KNeighborsClassifier(n_neighbors = 5)#Aqui nós colocamos os parâmetros que nós queremos mudar
# knn = knn.fit(x1_train, y1_train.values.ravel())#Criamos nosso objeto para o treino
# accuracyKnn = knn.score(x1_test, y1_test.values.ravel())

# print('Acurácia do KNN: ', accuracyKnn)

# #-----------------------------------------------------------------------------------------------------#

# #----------------------------------------------SVM----------------------------------------------------#

svm1 = svm.SVC(random_state = 2)#Aqui nós colocamos os parâmetros que nós queremos mudar
svm1 = svm1.fit(x1_train, y1_train.values.ravel())#Criamos nosso objeto para o treino
accuracySvm = svm1.score(x1_test, y1_test.values.ravel())

print('Acurácia do SVM Classifier: ',accuracySvm)
# #-----------------------------------------------------------------------------------------------------#

# #--------------------------------------------Rede Neural----------------------------------------------#

mlp = MLPClassifier(random_state = 2)#Aqui nós colocamos os parâmetros que nós queremos mudar
mlp = mlp.fit(x1_train, y1_train.values.ravel())#Criamos nosso objeto para o treino
accuracyMlp = mlp.score(x1_test, y1_test.values.ravel())

print('Acurácia da Rede neural Classifier: ', accuracyMlp)

# #-----------------------------------------------------------------------------------------------------#


# #------------------------------------------Random Forest----------------------------------------------#

randomForest = RandomForestClassifier()
randomForest.fit(x1_train, y1_train.values.ravel())
accuracyRandomForest = randomForest.score(x1_test, y1_test.values.ravel())

print('Acurácia do Random Forest Classifier: ', accuracyRandomForest)

# #-----------------------------------------------------------------------------------------------------#

cont1 = 1
cont2 = 0
pokemon1 = 0
pokemon2 = 0
change = False
combat = False

while(cont1):

	print("#------------------------------------------------------------#")
	print("#---------------------STATUS DO POKEMÓN----------------------#")
	print("#------------------------------------------------------------#")
	# Type1 = input("Type 1: ")
	# HP = input("HP: ")
	# Attack = input("Attack: ")
	# Defense = input("Defense: ")
	# Sp_attack = input("Sp Atk: ")
	# Sp_defense = input("Sp. Def: ")
	# Speed = input("Speed: ")
	# Legendary = input("Legendary: ")

	#Aqui nós criamos um novo dado para teste
	new_data = OrderedDict([
    	('Type 1', randint(0, 16)),
    	('HP', randint(5, 255)),
    	('Attack', randint(5, 255)),
    	('Defense', randint(5, 255)),
    	('Sp. Atk', randint(5, 255)),
    	('Sp. Def', randint(5, 255)),
    	('Speed', randint(5, 255)),
    	('Legendary', randint(0, 1))
    	])
	
	print(new_data)

	#Usamos esse novo dado para prever onde os status se encaixam
	#E fazemos as predições do possivel pokemon que pode ser
	new_data = pandasObj.Series(new_data).values.reshape(1,-1)
	# print(new_data)
	predict = tree1.predict(new_data)
	# time.sleep(60)
	pokemon_id = (predict - 1)
	pokemon_to_print = str(pokemon_name[pokemon_id])
	# print(predict)
	# print(pokemon_to_print)

	number_image = str(predict[0])
	print("#------------------------------------------------------------#")
	print("Id: ", number_image)

	remove = "['']"
	for i in range(0,len(remove)):
 		pokemon_to_print = pokemon_to_print.replace(remove[i],"")

	print("Pokémon: ", pokemon_to_print)
	print("#------------------------------------------------------------#")

	#Aqui é onde nós exibimos a "batalha" na tela
	if cont2 == 0:
		pokemon1 = (pokemon_id + 1)
		background = Image.open("Image/Grass_battle2.png").convert("RGBA")
		foreground = Image.open("Image/" + str(pokemon_to_print) + ".png").convert("RGBA")

		background.paste(foreground, (1000, 200), foreground)
		# showPIL(background)
		background.show()
		background.save('out.png')
		time.sleep(2)
		background.close()

	elif cont2 == 1:
		pokemon2 = (pokemon_id + 1)
		background = Image.open("out.png").convert("RGBA")
		foreground = Image.open("Image/" + str(pokemon_to_print) + ".png").convert("RGBA")

		background.paste(foreground, (200, 600), foreground)
		# showPIL(background)
		background.show()
		time.sleep(2)
		combat = True
		background.close()

	if cont2 == 1:
		cont2 = 0
	else:
		cont2+= 1

	if combat:
		print("-------------------------------------------")
		print("First_pokemon: ", pokemon1)
		print("Second_pokemon: ", pokemon2)
		if change == False:
			new_data_battle = OrderedDict([('First_pokemon', pokemon1), ('Second_pokemon', pokemon2)])
			change = True
		elif change == True:
			new_data_battle = OrderedDict([('First_pokemon', pokemon2), ('Second_pokemon', pokemon1)])
			change = False

		print("New_data_battle - 1: ", new_data_battle)
		new_data_battle = pandasObj.Series(new_data_battle).values.reshape(1,-1)
		print("New_data_battle - 2: ", new_data_battle)
		predict_battle = tree2.predict(new_data_battle)
		print("Predict_battle: ", new_data_battle)

		pokemon_id_Battle = (int(predict_battle)-1)
		print("Pokemon_id_battle: ", pokemon_id_Battle)
		pokemon_winner = str(pokemon_name[pokemon_id_Battle])
		print("Pokemon_Winner: ", pokemon_winner)

		remove = "['']"

		#Aqui exibimos quem será o ganhador da batalha
		for i in range(0,len(remove)):
 			pokemon_winner = pokemon_winner.replace(remove[i],"")
		print(pokemon_winner)
		background = Image.open("Image/Grass_battle.png").convert("RGBA")
		foreground = Image.open("Image/" + str(pokemon_winner) + ".png").convert("RGBA")

		background.paste(foreground, (600, 500), foreground)
		# showPIL(background)
		background.show()
		combat = False
		time.sleep(5)
