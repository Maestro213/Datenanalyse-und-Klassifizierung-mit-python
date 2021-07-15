"""
Created on Mon Jul  5 11:51:49 2021

@author: Kirill Gusev
"""
import numpy as np
import random 
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn import metrics


#Wholesale customers Data Set
data = pd.read_csv("/Users/fox2/Downloads/Wholesale_customers_data.csv")

print(data.shape)#Wie viel Reihen und Spalten hat importiertes DataFrame

print(data.sample(5))#Zufälige ausgabe von fünf Datenpunkten

print(data.info())#Gibt die Informaiton von Daten punkten in data

print(data.describe())#Statistische Auswertung
'''
#Plot die normiertn Daten
fig = plt.figure(figsize=(16,9))
data1=((data-data.min())/(data.max()-data.min())).iloc[:,2:]
print(data1)
data1.plot(kind='bar')
plt.locator_params(axis='x', nbins=20)
plt.title('Die relativen Anzahlen von gekaufte Wahren für jede Kunde')
plt.ylabel('normierte Anzahl')
fig.savefig('Visualizierung_den_Daten.png',dpi=100)

plt.show()
'''


#2.Variante: gibt es NaN Werte in data
print(data.isnull().sum())

'''
Die data Datensatz ist vollständig
Wenn es nicht vollständig wäre dann kann man NaN-Werte folgenderweisen handeln
data.dropna(how='any')
data.fillna(value=(data.mean()+random.uniform(data.mean()-data.std(),data.mean()+std())))

'''

print(data.columns) #Header

#Y - Klassen, X - Attribute
Y = data.iloc[:,0] 
X = data.iloc[:,1:]
Y[Y == 2] = 0#Ersetzen Werte, damit Klassen nur (0,1) enthalten

korr_ma = X.corr()
m = np.triu(korr_ma)#Dreieckige Matrizen
sn.heatmap(korr_ma,annot=True, mask=m)




def Korr_tab(data):
    col = list(data.columns)[1:]#Attributen Name
    d = []
    for i in col:
        for j in col[col.index(i)+1:]:
            d.append(i+' und '+j)
    names=np.array(d)
    korr_ma1 = m.flatten()
    korr = korr_ma1[(korr_ma1 !=0) & (korr_ma1!=1)]
    fig,ax = plt.subplots(figsize=(16,9))
    ax.barh(names,korr)
    ax.invert_yaxis()
    ax.set_title('Die Korrelationskoeffizienten zwischen Atributten')
    plt.show()

Korr_tab(data)

#Analysis von Wertebereich der Atributte
stat = X.describe()
Var = X.var()
print(stat,Var)

#Box Plot
boxplot = X.iloc[:,1:].boxplot(rot = 15, grid=True)

#Distribution Ploten
#Mit log-Scale
fig =plt.figure(figsize=(35,4))
i = 1
for c in X.columns:
    if c == 'Region':
        ax = plt.subplot(1,7,i)
        sn.histplot(X[c],discrete=True)
        tick = np.arange(len(X['Region'].unique())+1)[1:]
        plt.title('Distribution von Attribut ' + c)
        plt.xticks(tick,['Lisbon','Oporto','Other'])
        i +=1
    else:
        ax = plt.subplot(1,7,i)
        sn.histplot(X[c],log_scale=True)
        plt.title('Distribution von Attribut ' + c)
        i +=1
fig.savefig('Distribution_Log_Scale.png',dpi=100)
plt.show()
#Ohne Log-Scale
fig =plt.figure(figsize=(35,4))
i = 1
for c in X.columns:
    if c == 'Region':
        ax = plt.subplot(1,7,i)
        sn.histplot(X[c],discrete=True)
        tick = np.arange(len(X['Region'].unique())+1)[1:]
        plt.title('Distribution von Attribut ' + c)
        plt.xticks(tick,['Lisbon','Oporto','Other'])#Namen von Regionen
        i +=1
    else:
        ax = plt.subplot(1,7,i)
        sn.histplot(X[c],log_scale=None)
        plt.title('Distribution von Attribut ' + c)
        i +=1
fig.savefig('Distribution.png',dpi=100)
plt.show()

#Section 2

Y = data.iloc[:,0] 
X = data.iloc[:,1:]
Y[Y == 1] = 1
Y[Y == 2] = 0



X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=0)

klassifk_methoden = [LogisticRegression(),SVC(),DecisionTreeClassifier()]
klassifk_methoden_namen = ['Logistic Regression','Support Vector Machine','Dicision Tree']
tab = []

fig =plt.figure(figsize=(27,9))
i = 1
for n,m in zip(klassifk_methoden_namen,klassifk_methoden):
    #Modelle erstellen
    modele = m
    modele.fit(X_train, y_train)
    modele_pred = modele.predict(X_test)
    accuracy = metrics.accuracy_score(modele_pred, y_test)
    precision = metrics.precision_score(modele_pred, y_test)
    cnf_matrix = metrics.confusion_matrix(modele_pred, y_test)
    tab.append([accuracy,precision])
    #ploting
    ax = plt.subplot(1,3,i)
    klass_namen=['Retail','Horeca'] # name  of classes
    t = np.arange(len(klass_namen)+1)[1:]-0.5
    
    #heatmap
    sn.heatmap(pd.DataFrame(cnf_matrix), annot=True,cmap='crest',fmt='g')
    plt.xticks(t, klass_namen)
    plt.yticks(t, klass_namen)
    plt.tight_layout()
    plt.title('Confusion matrix für Klassifikationsmodelle - '+n, y=1.1)
    plt.ylabel('Tatsächlicher Wert')
    plt.xlabel('Vorhersagtee Wert')
    i +=1
fig.savefig('CNF_Matrizen.png')
tab = pd.DataFrame(tab,index =klassifk_methoden_namen ,columns=['Accuracy','Precision'])
print(tab)


#Section 3

'''
Backtesting
Veränder die Datensätze
'''
#Drop die (fast)konstatnte und nicht sinvolle Attribut - Regionen
new1_X = X.drop(['Region'],axis=1)

#Drop Attribut Grocery die stark korreliert mit Milk, Detergents_Paper
new2_X = X.drop(['Grocery'],axis=1)

#Normierter Datensatz
new3_X =((X-X.min())/(X.max()-X.min()))

#Etfernen die abnoralen Outliers
new_data = data
for n in list(data.columns):
    new_data = new_data[new_data[n] < (new_data[n].mean()+3*np.sqrt(new_data[n].var()))]
new4_X = new_data.iloc[:,1:]
new4_Y = new_data.iloc[:,0]
#new4_Y[new4_Y == 1] = 1
new4_Y[new4_Y == 2] = 0


#Erstellen die Modelle

data_mengen = [X,new1_X,new2_X,new3_X,new4_X]
namen = ['Originale Datensatz','DatenSatz 1','DatenSatz 2','DatenSatz 3','DatenSatz 4']
fig =plt.figure(figsize=(20,16))
i = 1
T= []
for d,na in zip(data_mengen,namen):#d -Datensätze, Name von dem Datensatz
    X = d
    tab=[]
    if len(d) == len(new4_X):
        Y = new4_Y
    X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=0)
    for n,m in zip(klassifk_methoden_namen,klassifk_methoden):
        modele = m
        modele.fit(X_train, y_train)
        modele_pred = modele.predict(X_test)
        accuracy = metrics.accuracy_score(modele_pred, y_test)
        precision = metrics.precision_score(modele_pred, y_test)
        cnf_matrix = metrics.confusion_matrix(modele_pred, y_test)
        tab.append([accuracy,precision])#Speichern die Accuracy und Precision Measures
    #plot
        ax = plt.subplot(5,3,i)
        klass_namen=['Retail','Horeca'] # Namen  von Klassen
        t = np.arange(len(klass_namen)+1)[1:]-0.5
        
        # heatmap
        sn.heatmap(pd.DataFrame(cnf_matrix), annot=True,cmap ='crest' ,fmt='g')
        plt.xticks(t, klass_namen)
        plt.yticks(t, klass_namen)
        plt.tight_layout()
        if i <=3:
            plt.title('Confusion Matrix für Klassifikationsmodelle - '+n, y=1.1)#Erzeugen Spaltensnamen
        if i%3 ==1:
            plt.ylabel(na+'\n'+' Tatsächlicher Wert')#Erzeugen Zielensnamen
        else:
            plt.ylabel('Tatsächlicher Wert')
        plt.xlabel('Vorhersagter Wert')
        i +=1
   
    T.append(pd.DataFrame(tab,index =klassifk_methoden_namen ,columns=['Accuracy_'+str(namen.index(na)),'Precision_'+str(namen.index(na))]))
fig.savefig('CNF_matrizen_Backtestet.png')
Table = pd.concat([T[0],T[1],T[2],T[3],T[4]],axis=1)
print(Table)