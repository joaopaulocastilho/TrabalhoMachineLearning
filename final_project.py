import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

def discretiza(data, c):
    classes = np.unique(data[0:,c])
    for i in range(0, len(classes)):
        data[0:,c][data[0:, c] == classes[i]] = i
    return data

#Ler dados
f = open('diabetic_data.csv')
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

#Tratamento dos dados faltantes
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

#Passo 2: Conjunto de Hiper Parametros
param_grid = {
    "criterion": ['gini', 'entropy'],
    "splitter" : ['best', 'random'],
    "class_weight" : ['balanced', None]
    #"max_features" : ["auto", None]
    #"presort" : [False, True]
}

#Passo 3: melhor combinacao de parametros para os 3 classificadores
gs = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'accuracy', cv = 5)
gs.fit(XTrain, YTrain)
YHat = gs.predict(XTest)

print("\n\n")
print(classification_report(YHat, YTest))
print("score: {:.2f}%".format(gs.best_score_ * 100))
print("\nMelhor Combinacao de parametros:")
print(gs.best_params_)
