# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Predicción de la TRM usando un modelo LSTM
# 
# ### Author:
# Johny Vallejo-Sánchez
# 
# *javallejos@eafit.edu.co*
# 
# Universidad EAFIT
# 
# ### Materia:
# Inteligencia Computacional Semestre 2020-01
# 
# ### Profesor:
# Santiago Medina PhD.
# %% [markdown]
# ## Resumen
# 
# Este Notebook contiene el desarrollo de un modelo **inteligencia artificial** implementado en Python, usando la API de Keras, para predecir el valor de la TRM diaria usando datos históricos desde el año 1996 hasta 2019. Para ello se implementó una arquitectura de **Redes Neuronales Recurrentes** (RNN), usando un tipo especial de éstas conocido como Long Short Term Memory (LSTM). Se realizó un análisis descriptivo y exploratorio de los datos antes de su procesamiento, luego se agregó variables derivadas al dataset y se preprocesó los datos para la ingesta en el modelo. Se implementó 3 capas, la primera (LSTM) con 180 neuronas, la segunda un dropot para mejorar el backpropagation, y la tercera (dense) con 1 neurona que entrega la salida, almacenando los pesos de la red en un archivo en formato hdf5. Finalmente se evaluó el modelo tanto en training como en testing, obteniendo un error de predicción de $1.99 COP.
# %% [markdown]
# ## Carga de librerías
# 
# A continuación se realiza la carga de las librerías necesarias para los cálculos y procesamiento de los datos:
# 
# + **Numpy**: Manejo de matrices y vectores
# + **Pandas**: Procesamiento de dataset
# + **Matplotlib**: Gráficos
# + **Seaborn**: Estimadores estadísticos
# + **Sklearn**: Pre-procesamiento de datos y métricas del modelo
# + **Keras**: Arquitectura de RNN

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error

# %% [markdown]
# ## Carga de datos
# 
# Los datos están disponibles en un repositorio en [GitHub](https://github.com/Hausdorff94/TRM-USD-Inteligencia-Computacional) y se leen de manera remota. A continuación se muestra los primeros registros del dataset cargado.

# %%
my_df = pd.read_csv('https://raw.githubusercontent.com/Hausdorff94/TRM-USD-Inteligencia-Computacional/master/data/dataTRM.csv')
my_df['Date'] = pd.to_datetime(my_df.Date).dt.date
my_df.head()


# %%
my_df.tail()

# %% [markdown]
# El dataset en el que se trabajará cuenta con las siguientes variables:
# 
# + **Date**: Fecha del registro
# + **infl**: Índice de inflación
# + **smlv**: Salario mínimo mensual vigente
# + **pob**: Número de habitantes
# + **Inverargos t**: Precio de la acción Argos
# + **BCOLOM t**: Precio acción Bancolombia
# + **EUR t**: Euro/Dolar
# + **TRM t**: Valor del dolar en COP
# + **TRM t-1**: Resago-1
# + **TRM t-2**: Resago-2
# + **TRM t+1**: Valor a predecir

# %%
df = my_df.copy()
df.count()

# %% [markdown]
# El dataset cuenta con 6.126 registros, desde el 5 de enero de 1996 hasta el 31 de diciembre de 2019.
# 
# 
# 
# Ahora se reorganiza el dataset renombrando las columnas e indicando la fecha en su respectivo formato y fijarla como índice. Ahora la **variable explicativa** se llama **trm-plus-1**.

# %%
# Rename columns
df.rename(columns={'Date' : 'date', 'Inverargos t' : 'inverargos', 'BCOLOM t' : 'bcolom', 
                   'EUR t' : 'eur', 'TRM t' : 'trm', 'TRM t-1' : 'trm-1', 'TRM t-2' : 'trm-2', 'TRM t+1':'trm-plus-1'}, inplace=True)
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
df.set_index('date', inplace=True)
df = df.astype(float)
df.head()

# %% [markdown]
# ## Serie de tiempo de la TRM
# 
# Observamos la serie de tiempo completa en la primera imagen, y en la segunda un zoom del último año, 2019.

# %%
sns.set(rc={'figure.figsize':(16, 8)}, palette='muted', style = "whitegrid")
df.plot(y='trm')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Fecha', fontsize = 20)
plt.ylabel('COP', fontsize = 20)
plt.title('Serie de tiempo TRM', fontsize = 25)
plt.legend(fontsize = 14)
plt.show()


# %%
df.loc['2019', 'trm'].plot()

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Fecha', fontsize = 20)
plt.ylabel('COP', fontsize = 20)
plt.title('Serie de tiempo TRM 2019', fontsize = 25)
plt.legend(fontsize = 14)
plt.show()

# %% [markdown]
# ## Análisis de correlación entre las variables
# 
# Para ver el impacto y correlación que existe entre las variables del dataset, usamos la correlación de Pearson y vemos cómo la variable de salida está correlacionada con las demás.

# %%
colormap = plt.cm.inferno
plt.figure(figsize=(15,15))
plt.title('Correlación de Pearson', y=1.05, size=15)
sns.heatmap(df.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()

plt.figure(figsize=(15,5))
corr = df.corr()
sns.heatmap(corr[corr.index == 'trm-plus-1'], linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True);

# %% [markdown]
# Aquí encontramos algo interesante, la variable *infl* que corresponde al índice de **inflación** anual de país está inversamente correlacionado con el valor de la TRM, lo cual tiene sentido, pues en el país se comercializan muchos productos que son importados, los que se ven afectados directamente por el dolar; al variar la inflación, el valor del producto también fluctuará en su precio en dólares, dándole una explicación al precio del dolar según el índice de inflación.
# 
# Por otro lado, vemos que las variables con rezago no tienen una correlación directa con la variable de salida, pero sí tiene una correlación más fuerte las variables de *salario mínimo mensual* y *el número de habitantes en el país*.
# %% [markdown]
# ## Función para predecir a $n$ días a futuro.
# 
# Se crea una función para configurar el dataset que haga el forecast para el siguiente día (por default es 1 día, en el argumento *look_back*).

# %%
def create_dataset(dataset, look_back=5):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# %% [markdown]
# ## Ranking de las variables por importancia en el 
# 
# Usamos un *Random Forest Regressor* para estimar el peso de las variables involucradas en el modelo.

# %%
# Scale and create datasets
target_index = df.columns.tolist().index('trm-plus-1')
dataset = df.values.astype('float32')

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Set look_back to 1 for 1 day to future 
X, y = create_dataset(dataset, look_back=1)
y = y[:,target_index]
X = np.reshape(X, (X.shape[0], X.shape[2]))


# %%
forest = RandomForestRegressor(n_estimators = 100)
forest = forest.fit(X, y)


# %%
importances = forest.feature_importances_
std = np.std([forest.feature_importances_ for forest in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

column_list = df.columns.tolist()
print("Ranking de cada variable:")
for f in range(X.shape[1]-1):
    print("%d. %s %d (%f)" % (f, column_list[indices[f]], indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(16,7))
plt.title("Importancia", fontsize = 20)
plt.bar(range(X.shape[1]), importances[indices],
       color="salmon", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices, fontsize = 16)
plt.yticks(fontsize = 16)
plt.xlim([-1, X.shape[1]])
plt.show()

# %% [markdown]
# ## Preprocesamiento de los datos
# 
# Aquí definimos la variable de salida y las variables explicativas en forma de matriz (array en Python). Con la librería *Scikit-learn*, escalamos los datos entre 0 y 1, y configuramos el dataset para ser ingresado directamente al modelo. Tomamos un dataset de Training y el otro de Testing, en una relación 85% y 15%, respectivamente. Se toma el 15% del final del dataset, ordenado por fecha, para no romper la serie de tiempo y tener continuidad de los datos en el tiempo.

# %%
# Scale and create datasets
target_index = df.columns.tolist().index('trm-plus-1')
dataset = df.values.astype('float32')

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Create y_scaler to inverse it later
y_scaler = MinMaxScaler(feature_range=(0, 1))
t_y = df['trm-plus-1'].values.astype('float32')
t_y = np.reshape(t_y, (-1, 1))
y_scaler = y_scaler.fit(t_y)
    
# Set look_back to 20 which is 5 hours (15min*20)
X, y = create_dataset(dataset, look_back=5)
y = y[:,target_index]

# Set training data size
train_size = int(len(X) * 0.85)
trainX = X[:train_size]
trainY = y[:train_size]
testX = X[train_size:]
testY = y[train_size:]

# %% [markdown]
# ## Creación de la RNN tipo LSTM
# 
# Con la API Keras, se crea la red LSTM con la siguiente configuración:
# 
# + **Capa 1**: LSTM con 180 neuronas.
# + **Capa 2**: Dropout que regulariza la red durante el entrenamiento, ignorando de manera random algunas neuronas, mejorando el backpropagation.
# + **Capa 3**: Dense, se encarga de la multiplicación matricial para dar el valor de salida de la red.
# 
# Sólo la capa 1 tiene una función de activación, que por default es la *tanh*.

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, LSTM, Dense

# create a small LSTM network
model = Sequential()
model.add(LSTM(180, input_shape=(X.shape[1], X.shape[2]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, kernel_initializer='random_normal'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse'])
print(model.summary())

# %% [markdown]
# El **número total de parámetros es 137.701**, lo que implicará un costo computacional alto. En la tabla anterior podemos ver la arquitectura de la red LSTM.
# %% [markdown]
# ## Training del modelo
# 
# Se entrena el modelo con la red propuesta anteriormente y se guarda los pesos calculados de la red en el archivo *weights.best.hdf5*, para ser usados más tarde en las validaciones del modelo y también se usa en un posible despliegue en producción. Se toma como métrica el **error cuadrático medio**, MSE, que más adelante lo podemos convertir en **RMSE**. Adicional a ello se tiene la métrica del **error absoluto promedio**, MAE.

# %%
# Save the best weight during training.
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint("weights.best.hdf5", monitor='val_mean_squared_error', verbose=1, save_best_only=True, mode='min')

# Fit
callbacks_list = [checkpoint]
history = model.fit(trainX, trainY, epochs=2500, batch_size=800, verbose=0, callbacks=callbacks_list, validation_split=0.1)

# %% [markdown]
# ## Performance en el entrenamiento
# 
# En los siguientes gráficos vemos que en Test como en Training, hay convergencia con cuando el número de Epochs aumenta:
# 
# + **Epochs**: 2500
# + + **Tamaño del batch**: 800

# %%
epoch = len(history.history['loss'])
for k in list(history.history.keys()):
    if 'val' not in k:
        plt.figure(figsize=(40,10))
        plt.plot(history.history[k])
        plt.plot(history.history['val_' + k])
        plt.title(k, fontsize = 34)
        plt.ylabel(k, fontsize = 27)
        plt.yticks(fontsize = 26)
        plt.xticks(fontsize = 26)
        plt.xlabel('epoch', fontsize = 27)
        plt.legend(['train', 'test'], loc='upper left', fontsize = 14)
        plt.show()

# %% [markdown]
# ### MSE en el entrenamiento

# %%
min(history.history['val_mean_absolute_error'])

# %% [markdown]
# ## Reentrenamiento del modelo
# 
# Tomando los mejores valores para los pesos de la red, se reentrena el modelo para mejorar el MSE

# %%
# Baby the model a bit
# Load the weight that worked the best
model.load_weights("weights.best.hdf5")

# Train again with decaying learning rate
from keras.callbacks import LearningRateScheduler
import keras.backend as K

def scheduler(epoch):
    if epoch%2==0 and epoch!=0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*.9)
        print("lr changed to {}".format(lr*.9))
    return K.get_value(model.optimizer.lr)
lr_decay = LearningRateScheduler(scheduler)

callbacks_list = [checkpoint, lr_decay]
history = model.fit(trainX, trainY, epochs=int(epoch/3), batch_size=500, verbose=0, callbacks=callbacks_list, validation_split=0.1)

# %% [markdown]
# ## Performance del modelo reentrenado.
# 
# Los gráficos de performance no muestran mejora sustancial entre el modelo reentrenado y el que ya se tenía, por lo que no valdría la pena un reentrenamiento en este caso.

# %%
epoch = len(history.history['loss'])
for k in list(history.history.keys()):
    if 'val' not in k:
        plt.figure(figsize=(40,10))
        plt.plot(history.history[k])
        plt.plot(history.history['val_' + k])
        plt.title(k, fontsize = 34)
        plt.ylabel(k, fontsize = 27)
        plt.yticks(fontsize = 26)
        plt.xticks(fontsize = 26)
        plt.xlabel('epoch', fontsize = 27)
        plt.legend(['train', 'test'], loc='upper left', fontsize = 14)
        plt.show()

# %% [markdown]
# # Visualización de las predicciones en Test
# 
# Aquí se compara el dataset de Testing con el pronosticado. Los datos se encuentran aún escalados.

# %%
# Benchmark
model.load_weights("weights.best.hdf5")

pred = model.predict(testX)

predictions = pd.DataFrame()
predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))
predictions['actual'] = testY
predictions = predictions.astype(float)

predictions.plot(figsize=(20,10))
plt.title('TRM en Testing',fontsize = 34)
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
plt.legend(loc='upper left', fontsize = 14)
plt.show()

# %% [markdown]
# ## Distribución del error en testing
# 
# Podemos ver que el error se distribuye normalmente, con media aproximadamente cero, lo que por el teorema del límite central, el error es **independiente e identicamente distribuido**.

# %%
predictions['diff'] = predictions['predicted'] - predictions['actual']
plt.figure(figsize=(10,10))
sns.distplot(predictions['diff']);
plt.title('Distribution of differences between actual and prediction')
plt.show()

print("RMSE : ", np.sqrt(mean_squared_error(predictions['predicted'].values, predictions['actual'].values)))
print("MAE : ", mean_absolute_error(predictions['predicted'].values, predictions['actual'].values))
predictions['diff'].describe()

# %% [markdown]
# # Visualización de la predicción como serie de tiempo
# 
# Se "desescalan" los datos y se grafican la serie de tiempo comparativa del valor real (azul) y el forecast (verde).

# %%
pred = model.predict(testX)
pred = y_scaler.inverse_transform(pred)
trm = y_scaler.inverse_transform(np.reshape(testY, (testY.shape[0], 1)))
predictions = pd.DataFrame()
predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))
predictions['trm-plus-1'] = pd.Series(np.reshape(trm, (trm.shape[0])))

p = df[-pred.shape[0]:].copy()
predictions.index = p.index
predictions = predictions.astype(float)

ax = predictions.plot(x=predictions.index, y='trm-plus-1', figsize=(40,10))
ax = predictions.plot(x=predictions.index, y='predicted', figsize=(40,10), ax=ax)
index = [str(item) for item in predictions.index]

plt.title('Predicción vs Actual',fontsize = 34)
plt.yticks(fontsize = 26)
plt.xticks(fontsize = 26)
plt.legend(loc='upper left', fontsize = 24)
plt.show()

# %% [markdown]
# ## Correlación entre el error y la TRM predicha
# 
# Se realiza un análisis de correlación entre el error de la predicción y el valor de la predicción y determinar si tienen relación dependiente significativa.
# 
# Claramente vemos que el error distribuye normal y tenemos un coeficiente de correlación de **0.45**, una correlación **débil** y lo corroboramos con el *p-value* **= 6.7e-47**, lo que nos indica que el valor de la correlación es estadísticamente significativo y podemos fiarnos de él.

# %%
pred = model.predict(testX)
pred = y_scaler.inverse_transform(pred)
trm = y_scaler.inverse_transform(np.reshape(testY, (testY.shape[0], 1)))
predictions = pd.DataFrame()
predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))
predictions['trm-plus-1'] = pd.Series(np.reshape(trm, (trm.shape[0])))
predictions['diff'] = predictions['predicted'] - predictions['trm-plus-1']
g = sns.jointplot("diff", "predicted", data=predictions, kind="kde", space=0)
plt.title('Distribución del error y TRM predecida')
plt.show()

# %% [markdown]
# ## RMSE
# 
# Finalmente, encontramos que las métricas para nuestro modelo son las siguientes:
# 
# + **RMSE** = 1.98 COP
# + **MAE** = 1.59 COP

# %%
print("RMSE : ", np.sqrt(mean_squared_error(predictions['predicted'].values, predictions['trm-plus-1'].values)))
print("MAE : ", mean_absolute_error(predictions['predicted'].values, predictions['trm-plus-1'].values))
#predictions['diff'].describe()

# %% [markdown]
# ## Conclusiones
# 
# + La inclusión de variables externas, como el índice de inflación, es fundamental para darle explicatividad al modelo que estamos construyendo.
# + Realizar un análisis descriptivo y exploratorio es lo más importante al momento de crear un modelo de predicción, pues nos da idea de si hay valores faltantes, outliers o si es necesario incluir más variables.
# + Las redes neuronales recurrentes RNN del tipo LSTM son apropiadas para predecir variables de salida numéricas que dependen de datos del pasado o pueden repetir tendencias, como el caso de la TRM.
# %% [markdown]
# # Resumen del modelo
# | Parámetros| RMSE   |MAE  |Modelo|
# |-----------|--------|-----|------|
# |   137,701 | 1.99   |1.56 |.hdf5 |

# %%


