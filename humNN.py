import tensorflow as tf
import pandas as pd
import numpy as np


__all__ = ["predict_humidity"]


data_path = "data/humidity.csv"

starting_dataset = pd.read_csv(data_path)

cols = ['month', 'day', 'hour', 'minute', 'recnt_Humidity', 'recnt_Temperature', 'Target_Humidity']

model = None

def create_dataset(X, y, n_samples):
    X_res, y_res = [], []
    for i in range(len(X) - n_samples):
        v = X.iloc[i:(i + n_samples)].values
        X_res.append(v)
        y_res.append(y.iloc[i + n_samples])
    return np.array(X_res), np.array(y_res)



"""
Let's start by loading the dataset into a Pandas Dataframe. 'Year' and 'Second' columns are removed since they've the
same value for each datapoint. 
"""

data = starting_dataset[cols]

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

target_hum_mean = x_train['Target_Humidity'].mean(axis=0)
target_hum_std = x_train['Target_Humidity'].std(axis=0)

x_train['norm_recnt_hum'] = (x_train['recnt_Humidity'] - recnt_hum_mean) / recnt_hum_std
x_train['norm_recnt_temp'] = (x_train['recnt_Temperature'] - recnt_temp_mean) / recnt_temp_std
x_train['norm_target_hum'] = (x_train['Target_Humidity'] - target_hum_mean) / target_hum_std

x_test['norm_recnt_hum'] = (x_test['recnt_Humidity'] - recnt_hum_mean) / recnt_hum_std
x_test['norm_recnt_temp'] = (x_test['recnt_Temperature'] - recnt_temp_mean) / recnt_temp_std
x_test['norm_target_hum'] = (x_test['Target_Humidity'] - target_hum_mean) / target_hum_std

"""
Let's select the columns needed for the neural network
"""

features_cols = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos',
                 'minute_sin', 'minute_cos', 'norm_recnt_hum', 'norm_recnt_temp']
prediction_col = ['norm_target_hum']

train_features = x_train[features_cols]
train_label = x_train[prediction_col]

test_features = x_test[features_cols]
test_label = x_test[prediction_col]

"""
Now let's create the neural network.
"""

n_samples = 36

train_x, train_y = create_dataset(train_features, train_label, n_samples)

test_x, test_y = create_dataset(test_features, test_label, n_samples)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=16, dropout=0.6, recurrent_dropout=0.6))
model.add(tf.keras.layers.Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

history = model.fit(train_x, train_y, batch_size=128, validation_split=0.15, shuffle=False, epochs=40)

test_loss, test_accuracy = model.evaluate(test_x,test_y)
print("Loss of the model: "+str(test_loss))
print("temp std: " + str(target_hum_std))
print("temp mean: " + str(target_hum_mean))
model.summary()
# use model.predict(data) to make a prediction


def normalize_dataset(data):
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12.0)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12.0)

    data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31.0)
    data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31.0)

    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 23.0)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 23.0)

    data['minute_sin'] = np.sin(2 * np.pi * data['minute'] / 50.0)
    data['minute_cos'] = np.cos(2 * np.pi * data['minute'] / 50.0)

    data['norm_recnt_hum'] = (data['recnt_Humidity'] - recnt_hum_mean) / recnt_hum_std
    data['norm_recnt_temp'] = (data['recnt_Temperature'] - recnt_temp_mean) / recnt_temp_std
    data['norm_target_hum'] = (data['Target_Humidity'] - target_hum_mean) / target_hum_std


def predict_humidity(rows):
    if model:
        df = pd.DataFrame(rows, columns=cols)
        df = pd.DataFrame(np.array(rows), columns=cols)
        normalize_dataset(df)
        df = df[features_cols]

        prediction = model.predict(np.array([df.to_numpy()]))
        return (float(prediction[0][0]) * target_hum_std) + target_hum_mean 