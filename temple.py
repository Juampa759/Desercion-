import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Bancos.csv')
x = dataset.iloc[:, 3:13].values 
y = dataset.iloc[:,13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
x[:,1] = labelencoder_x_1.fit_transform(x[:,1])
labelencoder_x_2 = LabelEncoder()
x[:,2] = labelencoder_x_2.fit_transform(x[:,2])

from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [2])], remainder='passthrough')
x = np.array(columnTransformer.fit_transform(x), dtype = np.float)




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#importando librerias y paquetes de keras
import keras 
from keras.models import Sequential
from keras.layers import Dense

#inicializando red neuronal
clasificador = Sequential()

#Agregando capas input y primera capa oculta
clasificador.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
#agredando segunda capa oculta
clasificador.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
#agregando capa de salida
clasificador.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

#Compilando red neuronal / Desenso Gradiente estocastico
clasificador.compile(optimizer= 'adam', loss= 'binary_crossentropy',metrics=['accuracy'] )

#ajustando red neuronal en el set de entrenamiento 
clasificador.fit(x_train, y_train, batch_size=10, epochs=100)

# Prediciendo Set de Prueba
y_pred = clasificador.predict(x_test)
y_pred = (y_pred>0.5)

# Matriz de Confusion
from sklearn.metrics import confusion_matrix
mc = confusion_matrix(y_test,y_pred)

#probando la red neuronal
"""
Geografía: Francia
Puntaje de crédito: 600
Género: Masculino
Edad: 40 años
Tenencia: 3 años
Saldo: $60000
Número de productos: 2
¿Tiene este cliente una tarjeta de crédito? Sí
¿Este cliente es un miembro activo? Si
Salario estimado: $50000
"""

nuevo_cliente = clasificador.predict(sc.transform(np.array([[0,1,600,0,40,3,60000,2,1,1,50000]])))
nuevo_cliente = (nuevo_cliente>0.5)
