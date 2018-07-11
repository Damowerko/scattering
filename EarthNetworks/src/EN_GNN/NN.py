
# coding: utf-8

# In[2]:


#!/usr/bin/env python3

region = "NYC"
n_stations = 25
grid_search = True
scale = True
categorical = False
fields = ["PressureSeaLevelMBar"
          ,"TemperatureC"
          ,"WindSpeedKph"
          ,"PressureSeaLevelMBarRatePerHour"
          ,"Humidity"
          ,"HumidityRatePerHour"
          ,"RainMillimetersRatePerHour"
]

print("""
Region: {}
N stations: {}
Normalize data: {}
fields: {}
""".format(region, n_stations, scale, fields))


# In[3]:


import sys, os
sys.path.insert(0, '..')

from EN_GNN.graph import get_distance_graph
from EN_GNN.data import pick_greedy, import_data
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

N = n_stations*len(fields)
stations = pick_greedy(region, n_stations=n_stations)
data, labels = import_data(fields, stations, region)  # type: (np.ndarray, np.ndarray)
data = data.astype(dtype=np.float32)

if scale:
    minimum = data.min(0).min(0)
    minimum = np.tile(minimum, (data.shape[0], n_stations,1))
    maximum = data.max(0).max(0)
    maximum = np.tile(maximum, (data.shape[0], n_stations,1))
    data = (data - minimum) / maximum

X = data.reshape(-1, N)
labels = labels.astype(dtype=np.int32).flatten()
y = labels

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)
y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)


# In[4]:


neg_weight = 1 / (X_train.shape[0] - np.sum(y_train))
pos_weight = 1 / (np.sum(y_train))

#neg_weight = 1
#pos_weight = 1
sample_weights = (1-y_train)*neg_weight + y_train*pos_weight
print("\nNegative weight: {} | Positive weight: {}\n".format(neg_weight,pos_weight))


# In[5]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.backend as K
from sklearn.model_selection import GridSearchCV, train_test_split
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import f1_score, accuracy_score

#def f1(y_true, y_pred):
#    positives = K.sum(y_true)
#    true_positives = K.dot(K.transpose(y_true),y_pred)
#    false_positives = K.dot(K.transpose(K.ones(K.shape(y_true)) - y_true), y_pred)
#    precision = true_positives / (true_positives + false_positives)
#    recall = true_positives / positives
#    return 2.0 * (recall*precision) / (recall + precision)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def create_model(M1=1000,
                 M2=None,
                 M3=None,
                 M4=None,
                 activation='relu', 
                 init_mode='uniform',
                 regularization=0, 
                 dropout=0.4, 
                 learning_rate=0.001, 
                 b1=0.9, 
                 b2=0.999):
    model = Sequential()
    model.add(Dense(M1, input_dim=N, kernel_initializer=init_mode))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
    if M2 is not None:
        model.add(Dense(M2, kernel_initializer=init_mode))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
    if M3 is not None:
        model.add(Dense(M3, kernel_initializer=init_mode))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
    if M4 is not None:
        model.add(Dense(M4, kernel_initializer=init_mode))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
    if categorical:
        model.add(Dense(2, activation='softmax'))
    else:
        model.add(Dense(1, activation='sigmoid'))
    adam = Adam(lr=learning_rate, beta_1=b1, beta_2=b2, epsilon=None, decay=0.0, amsgrad=False)
    if categorical:
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[f1,'accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[f1,'mse'])
    return model


# In[6]:


if not grid_search:
    model_params = dict(
        M1=1000,
        M2=1000,
        M3=1000,
        M4=1000,
        activation='relu', 
        init_mode='random_uniform',
        regularization=0, 
        dropout=0.5, 
        learning_rate=0.001, 
        b1=0.9, 
        b2=0.999
    )
    model = create_model(**model_params)
    if categorical:
        model.fit(X_train, y_train_cat, batch_size=100, epochs=100, validation_split=0.1, sample_weight=sample_weights)
        results = model.evaluate(X_test, y_test_cat)
    else:
        model.fit(X_train, y_train, batch_size=100, epochs=100, validation_split=0.1, sample_weight=sample_weights)
        results = model.evaluate(X_test, y_test)
    pred = model.predict(X_test)
    print("""
    -------------------------
    Parameters: {}

    Test Results
    -------------------------
    Loss: {r[0]}
    F1 Score: {r[1]}
    Accuracy: {r[2]}
    -------------------------
    """.format(model_params, r=results))


# In[ ]:


## Grid search
if grid_search:
    model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=100, verbose=0)
    param_grid = dict(
        M1=[500, 1000, 2000],
        M2=[None, 100, 500, 1000],
        #learning_rate=[0.001, 0.0005, 0.0001],
        dropout=[0.0, 0.4, 0.6]
    )
    grid_search = GridSearchCV(model, param_grid, scoring=['f1', 'accuracy'], refit='f1', verbose=10, n_jobs=8)
    print("Training model...")
    if categorical:
        grid_result = grid_search.fit(X_train, y_train_cat, sample_weight=sample_weights)
        pred = grid_search.predict(X_test)
    else:
        grid_result = grid_search.fit(X_train, y_train, sample_weight=sample_weights)
        pred = grid_search.predict(X_test)
    print(grid_result)
    f1 = f1_score(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    print("""
    Test Results
    ---------------
    F1 Score: {}
    Accuracy: {}
    ---------------
    """.format(f1, accuracy))


# In[ ]:


from sklearn.metrics import classification_report
if not grid_search:
    pred = model.predict(X_test)
else:
    pred = grid_search.predict(X_test)
if categorical:
    pred = np.argmax(pred, 1)
print(classification_report(y_test, pred, target_names=['No outage', 'outage']))

