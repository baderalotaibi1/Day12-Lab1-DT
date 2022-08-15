import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

#Q2: Read instagram_users.csv file

df=pd.read_csv('C:/Users/bader/OneDrive/gitlesson/Day12-Lab1-DT/instagram_users.csv')
print(df.head())
#Q3: Split tha dataset into training and testing

x=df.drop('real_fake',axis=1)
y=df['real_fake']
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.3)

#Q4.1: The first machine model

md1=DecisionTreeClassifier()
md1.fit(x_tr,y_tr)
predictions = md1.predict(x_ts)
print('model1:',md1,end="\n")
print('model 1 : \n',classification_report(y_ts,predictions),"\n")
print(accuracy_score(y_ts,predictions))
plot_confusion_matrix(md1, x_ts, y_ts)
plt.show()

#Q4.2: The second machine model

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_tr, y_tr)
rfc_pred = rfc.predict(x_ts)
print('model 2:',rfc,end="\n")
print('model 2 : \n',classification_report(y_ts,rfc_pred),"\n")
print(accuracy_score(y_ts,rfc_pred))
plot_confusion_matrix(rfc, x_ts, y_ts)
plt.show()
#Q4.3: The third machine model

rfc = RandomForestClassifier(n_estimators=250)
rfc.fit(x_tr, y_tr)
rfc_pred = rfc.predict(x_ts)
print('model 3:',rfc,end="\n")
print('model 3 : \n',classification_report(y_ts,rfc_pred),"\n")
print(accuracy_score(y_ts,rfc_pred))
plot_confusion_matrix(rfc, x_ts, y_ts)
plt.show()
