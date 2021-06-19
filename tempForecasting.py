import keras
import pandas as pd
import tensorflow as tf
import sklearn as sl
import numpy as np
import matplotlib.pyplot as plt

data_path = "C:\\Users\\franc\\PycharmProjects\\iotSystems\\data\\temperature.csv"

starting_dataset = pd.read_csv(data_path)

"""
A function that takes a dataset (X) and the corresponding labels (y). The output is a dataset (X_res): X_res[i] is a 
2D vector containing the n_samples samples preceding the label y_res[i+1].
These two objects will be used to train and test a neural network.
"""

def create_dataset(X, y, n_samples):
    X_res, y_res = [], []
    for i in range(len(X) - n_samples):
        v = X.iloc[i:(i + n_samples)].values
        X_res.append(v)
        y_res.append(y.iloc[i + n_samples])
    return np.array(X_res), np.array(y_res)


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

"""
Let's start with the preprocessing. 
The 'year' and 'second' columns will be removed since each of them has the same value for each row 
(dataset related only to year 2020, temperature sensed every 10 minutes).

"""

print(starting_dataset.head())

cols = ['month', 'day', 'hour', 'minute', 'recnt_Humidity', 'recnt_Temperature', 'Target_Temperature']

data = starting_dataset[cols]

print(data.head())
print(np.unique(data['month']))
print(np.unique(data['minute']))

"""
Now we've to normalize the columns in order to make them suitable as input to a neural network.
The columns 'month', 'day', 'hour' and 'minute' cannot be processed in the same way as the others, 
because their values are cyclic. By using the following approach, very common in practice, 
we'll let the NN understand that, for example, hours 23 and 00 have difference of 1 and not of -23.
"""

data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12.0)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12.0)

data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31.0)
data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31.0)

data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 23.0)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 23.0)

data['minute_sin'] = np.sin(2 * np.pi * data['minute'] / 50.0)
data['minute_cos'] = np.cos(2 * np.pi * data['minute'] / 50.0)

print(data.head())

print(data['recnt_Humidity'].mean(axis=0))
print(data['recnt_Humidity'].std(axis=0))

"""
Now let's normalize the three remaining columns and make each of them having mean == 0 and std_deviation == 1.
"""

train_percentage = 0.8

x_train = data.iloc[0:int(len(data)*train_percentage)]   # training set
x_test = data.iloc[int(len(data)*train_percentage):]     # test set

recnt_hum_mean = x_train['recnt_Humidity'].mean(axis=0)
recnt_hum_std = x_train['recnt_Humidity'].std(axis=0)

recnt_temp_mean = x_train['recnt_Temperature'].mean(axis=0)
recnt_temp_std = x_train['recnt_Temperature'].std(axis=0)

target_temp_mean = x_train['Target_Temperature'].mean(axis=0)
target_temp_std = x_train['Target_Temperature'].std(axis=0)

x_train['norm_recnt_hum'] = (x_train['recnt_Humidity'] - recnt_hum_mean) / recnt_hum_std
x_train['norm_recnt_temp'] = (x_train['recnt_Temperature'] - recnt_temp_mean) / recnt_temp_std
x_train['norm_target_temp'] = (x_train['Target_Temperature'] - target_temp_mean) / target_temp_std

x_test['norm_recnt_hum'] = (x_test['recnt_Humidity'] - recnt_hum_mean) / recnt_hum_std
x_test['norm_recnt_temp'] = (x_test['recnt_Temperature'] - recnt_temp_mean) / recnt_temp_std
x_test['norm_target_temp'] = (x_test['Target_Temperature'] - target_temp_mean) / target_temp_std

"""
Let's select the columns needed for the neural network
"""

features_cols = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos',
                 'minute_sin', 'minute_cos', 'norm_recnt_hum', 'norm_recnt_temp']
prediction_col = ['norm_target_temp']

train_features = x_train[features_cols]
train_label = x_train[prediction_col]

test_features = x_test[features_cols]
test_label = x_test[prediction_col]

print("Printing training set \n")
print(train_features.head())
print("\nPrinting training label \n")
print(train_label.head())

print("Printing test set \n")
print(test_features.head())
print("\nPrinting test label \n")
print(test_label.head())

"""
Now we can create and train the neural network. We'll try different configurations in terms of number of samples to use 
for prediction, number of layers, number of units of each layer, and so on.
Since we've to process a time series, an LSTM network will be used. 
"""
"""
samples_values = [12, 18, 24, 30, 36]
units_values = [32, 64, 128]

for samples in samples_values:
    
    print("Current number of samples: " + str(samples))
    
    train_x, train_y = create_dataset(train_features, train_label, samples)

    test_x, test_y = create_dataset(test_features, test_label, samples)

    for units in units_values:
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=units, dropout=0.))

"""

samples = [12, 24, 36, 60]
"""
train_x, train_y = create_dataset(train_features, train_label, samples[0])

test_x, test_y = create_dataset(test_features, test_label, samples[0])

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=32, dropout=0.5, recurrent_dropout=0.5))
model.add(tf.keras.layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

history = model.fit(train_x, train_y, batch_size=128, validation_split=0.15, shuffle=False, epochs=40)




#visualize_loss(history, "Training and Validation Loss")

test_loss, test_accuracy = model.evaluate(test_x, test_y)
print("loss of 1st model: "+str(test_loss))
"""
"""
Let's try to improve it by just increasing the number of samples to 18 and 24.
"""

train_x_2, train_y_2 = create_dataset(train_features, train_label, samples[1])

test_x_2, test_y_2 = create_dataset(test_features, test_label, samples[1])

model_2 = tf.keras.Sequential()
model_2.add(tf.keras.layers.LSTM(units=32, dropout=0.5, recurrent_dropout=0.5))
model_2.add(tf.keras.layers.Dense(1))

model_2.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

history_2 = model_2.fit(train_x_2, train_y_2, batch_size=128, validation_split=0.15, shuffle=False, epochs=40)

test_loss_2, test_accuracy_2 = model_2.evaluate(test_x_2,test_y_2)
print("loss of 2nd model: "+str(test_loss_2))

#visualize_loss(history_2, "Training and Validation Loss (2nd model)")

"""

train_x_3, train_y_3 = create_dataset(train_features, train_label, samples[2])

test_x_3, test_y_3 = create_dataset(test_features, test_label, samples[2])

model_3 = tf.keras.Sequential()
model_3.add(tf.keras.layers.LSTM(units=32, dropout=0.5, recurrent_dropout=0.5))
model_3.add(tf.keras.layers.Dense(1))

model_3.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

history_3 = model_3.fit(train_x_3, train_y_3, batch_size=128, validation_split=0.15, shuffle=False, epochs=40)

test_loss_3, test_accuracy_3 = model_3.evaluate(test_x_3,test_y_3)
print("loss of 3rd model: "+str(test_loss_3))

#visualize_loss(history_3, "Training and Validation Loss (3rd model)")


train_x_4, train_y_4 = create_dataset(train_features, train_label, samples[3])

test_x_4, test_y_4 = create_dataset(test_features, test_label, samples[3])

model_4 = tf.keras.Sequential()
model_4.add(tf.keras.layers.LSTM(units=32, dropout=0.5, recurrent_dropout=0.5))
model_4.add(tf.keras.layers.Dense(1))

model_4.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

history_4 = model_4.fit(train_x_4, train_y_4, batch_size=128, validation_split=0.15, shuffle=False, epochs=40)

test_loss_4, test_accuracy_4 = model_4.evaluate(test_x_4,test_y_4)
print("loss of 4th model: "+str(test_loss_4))

#visualize_loss(history_4, "Training and Validation Loss (4th model)")
"""
"""
From both the plots and the training log, we see that the four models achieve almost the same values of MSE.
During the different epochs, the 1st model never goes lower than 0.04, while the other three models reach MSE values 
around 0.036. So, between these four models, the second one represents the best choice since it has the best behavior
in terms of overfitting.
Let's now try to improve its MSE. We'll adopt two different strategies and see what's the best: on one hand, we keep 
the model as single-layer and increase the number of units of such layer, on the other hand we increase the number of 
layers while keeping as 32 the number of units.
"""
"""
model_2_64 = tf.keras.Sequential()
model_2_64.add(tf.keras.layers.LSTM(units=64, dropout=0.5, recurrent_dropout=0.5))
model_2_64.add(tf.keras.layers.Dense(1))

model_2_64.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

history_2_64 = model_2_64.fit(train_x_2, train_y_2, batch_size=128, validation_split=0.15, shuffle=False, epochs=40)

test_loss_2_64, test_accuracy_2_64 = model_2_64.evaluate(test_x_2,test_y_2)
print("loss of 2nd model with 64 units: "+str(test_loss_2_64))

#visualize_loss(history_2_64, "Training and Validation Loss (2nd model, 64 units, single layer)")


model_2_128 = tf.keras.Sequential()
model_2_128.add(tf.keras.layers.LSTM(units=128, dropout=0.5, recurrent_dropout=0.5))
model_2_128.add(tf.keras.layers.Dense(1))

model_2_128.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

history_2_128 = model_2_128.fit(train_x_2, train_y_2, batch_size=128, validation_split=0.15, shuffle=False, epochs=40)

test_loss_2_128, test_accuracy_2_128 = model_2_128.evaluate(test_x_2,test_y_2)
print("loss of 2nd model with 128 units: "+str(test_loss_2_128))

#visualize_loss(history_2_128, "Training and Validation Loss (2nd model, 128 units, single layer)")
"""
"""
Now we increase the number of layers.
"""

"""
model_2_2layers = tf.keras.Sequential()
model_2_2layers.add(tf.keras.layers.LSTM(units=32, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
model_2_2layers.add(tf.keras.layers.LSTM(units=32, dropout=0.5, recurrent_dropout=0.5))
model_2_2layers.add(tf.keras.layers.Dense(1))

model_2_2layers.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

history_2_2layers = model_2_2layers.fit(train_x_2, train_y_2, batch_size=128, validation_split=0.15, shuffle=False, epochs=40)

test_loss_2_2layers, test_accuracy_2_2layers = model_2_2layers.evaluate(test_x_2,test_y_2)
print("loss of 2nd model with two layers: "+str(test_loss_2_2layers))

#visualize_loss(history_2_2layers, "Training and Validation Loss (2nd model, 32 units, two layers)")




model_2_3layers = tf.keras.Sequential()
model_2_3layers.add(tf.keras.layers.LSTM(units=32, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
model_2_3layers.add(tf.keras.layers.LSTM(units=32, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
model_2_3layers.add(tf.keras.layers.LSTM(units=32, dropout=0.5, recurrent_dropout=0.5))
model_2_3layers.add(tf.keras.layers.Dense(1))

model_2_3layers.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

history_2_3layers = model_2_3layers.fit(train_x_2, train_y_2, batch_size=128, validation_split=0.15, shuffle=False, epochs=40)

test_loss_2_3layers, test_accuracy_2_3layers = model_2_3layers.evaluate(test_x_2,test_y_2)
print("loss of 2nd model with three layers: "+str(test_loss_2_3layers))

#visualize_loss(history_2_3layers, "Training and Validation Loss (2nd model, 32 units, three layers)")

"""
"""
Lets try to increase the number of epochs
"""

model_2_more_epochs = tf.keras.Sequential()
model_2_more_epochs.add(tf.keras.layers.LSTM(units=32, dropout=0.5, recurrent_dropout=0.5))
model_2_more_epochs.add(tf.keras.layers.Dense(1))

model_2_more_epochs.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

history_2_more_epochs = model_2_more_epochs.fit(train_x_2, train_y_2, batch_size=128, validation_split=0.15, shuffle=False, epochs=100)

test_loss_2_more_epochs, test_accuracy_2_more_epochs = model_2_more_epochs.evaluate(test_x_2,test_y_2)
print("loss of 2nd model with more epochs: "+str(test_loss_2_more_epochs))

visualize_loss(history_2_more_epochs, "Training and Validation Loss (2nd model, more epochs)")


























"""

data['norm_recnt_hum'] = (data['recnt_Humidity'] - data['recnt_Humidity'].mean(axis=0)) / data['recnt_Humidity'].std(axis = 0)
data['norm_recnt_temp'] = (data['recnt_Temperature'] - data['recnt_Temperature'].mean(axis=0)) / data['recnt_Temperature'].std(axis = 0)
data['norm_target_temp'] = (data['Target_Temperature'] - data['Target_Temperature'].mean(axis=0)) / data['Target_Temperature'].std(axis = 0)
print(data.head())
print(data[['recnt_Humidity', 'norm_recnt_hum','recnt_Temperature', 'norm_recnt_temp', 'Target_Temperature', 'norm_target_temp']])





print(len(data), len(x_train), len(x_test))

"""






