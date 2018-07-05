'''
Aluno: Joao Paulo Kubaszewski Castilho
Matricula: 1511100008

Primeiramente foi lido o dataset diabetic_data.csv, e trocado todos os ? por NaN (isso para poder usar a Imputer
e tratar os dados faltantes).

Depois os dados foram discretizados utilizando a funcao discretiza. 10 por cento dos dados foram para testes, enquanto 90 para treinamento.

Os algoritmos classificadores escolhidos foram: KNeighbors Classifier, Decision Tree Classifier e AdaBoost Classifier.

'''

import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model

def discretiza(data, c):
    classes = np.unique(data[0:,c])
    for i in range(0, len(classes)):
        data[0:,c][data[0:, c] == classes[i]] = i
    return data

#Ler dados
try:
    f = open('diabetic_data.csv')
except:
    print('Arquivo diabetic_data.csv nao encontrado')
    exit(0)
data = np.array([l.split(',') for l in f.readlines()])
data = data[1:,:]
f.close()

#Pegar todas as features com ? (dados faltantes) e trocar por np.nan
##Tem essa informacao no site https://www.hindawi.com/journals/bmri/2014/781670/tab1/
for i in range(0, data.shape[1]):
    data[:,i][data[:,i] == "?"] = np.nan

#Discretizar cada feature do dataset
naoDiscretiza = [0, 1, 6, 7, 8, 9, 13, 14, 15, 16, 17, 21, 50]

for i in range(0, data.shape[1]):
    if i not in naoDiscretiza:
        print("Discretizando feature", i)
        discretiza(data, i)

data = np.delete(data, (39, 40), axis = 1) #Colunas 39 e 40 sao constantes

#Tratamento dos dados faltantes utilizado o Imputer
imp = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
imp.fit(data)
imp.transform(data)


#Separando o dataset no conjunto X e y
X = np.array(data[:,:-1], dtype = float)
y = np.array([data[:,-1]], dtype = float).T


#Passos do trabalho:
#Passo 1: 10 melhores atributos para a criacao do modelo
K = 10
kbest = SelectKBest(k = K).fit(X, y.ravel())
features = kbest.get_support(indices = True)
X = kbest.fit_transform(X, y.ravel())

print("\n\nAs melhores 10 features sao:")
print(features)

#Passo 2: dividir o dataset
XTrain, XTest, YTrain, YTest = train_test_split(X, y, test_size = 0.1)

# ** Executando KNeighbors Classifier ********************************** #
print("\n\nExecutando KNeighbors Classifier...")

#Conjunto de Hiper Parametros para KNeighbors Classifier
KN_P = {
    "n_neighbors": [3, 4, 5],
    "weights" : ['uniform', 'distance'],
    "algorithm" : ['ball_tree', 'kd_tree']
}

#Melhor combinacao de parametros
gs = GridSearchCV(KNeighborsClassifier(), KN_P, scoring = 'accuracy', cv = 5)
gs.fit(XTrain, YTrain.ravel())
YHat = gs.predict(XTest)

print("\nMelhor Combinacao de parametros:")
print(gs.best_params_)

#classification_report

print("\nPerformance do Classificador:")
print(classification_report(YHat, YTest))
print("score: {:.2f}%".format(gs.best_score_ * 100))
print("----------------------------------------------------")
# ********************************** Executando KNeighbors Classifier ** #


# ** Executando Decision Tree Classifier ********************************** #
print("\n\nExecutando Decision Tree Classifier...")

#Conjunto de Hiper Parametros para Decision Tree Classifier
DT_P = {
    "criterion": ['gini', 'entropy'],
    "splitter" : ['best', 'random'],
    "class_weight" : ['balanced', None]
}

#Melhor combinacao de parametros
gs = GridSearchCV(DecisionTreeClassifier(), DT_P, scoring = 'accuracy', cv = 5)
gs.fit(XTrain, YTrain)
YHat = gs.predict(XTest)

print("\nMelhor Combinacao de parametros:")
print(gs.best_params_)

#classification_report

print("\nPerformance do Classificador:")
print(classification_report(YHat, YTest))
print("score: {:.2f}%".format(gs.best_score_ * 100))
print("----------------------------------------------------")
# ********************************** Executando Decision Tree Classifier ** #


# ** Executando AdaBoost Classifier ********************************** #
print("\n\nExecutanto AdaBoost Classifier (demora um tempo)...")

#Conjunto de Hiper Parametros para AdaBoost Classifier
AB_P = {
    "n_estimators" : [30, 40, 50],
    "learning_rate" : [0.7, 0.9],
    "algorithm" : ['SAMME', 'SAMME.R'],
}

#Melhor combinacao de parametros
gs = GridSearchCV(AdaBoostClassifier(), AB_P, scoring = 'accuracy', cv = 5)
gs.fit(XTrain, YTrain.ravel())
YHat = gs.predict(XTest)

print("\nMelhor Combinacao de parametros:")
print(gs.best_params_)

#classification_report

print("\nPerformance do classificador:")
print(classification_report(YHat, YTest))
print("score: {:.2f}%".format(gs.best_score_ * 100))
# ********************************** Executando AdaBoost Classifier ** #
