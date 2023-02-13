import data_processing_LSTM as D
import LSTM_model as L
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

import tensorflow.keras.utils as utils

seed = 5

utils.set_random_seed(
    seed
)

#get data from binance
data = D.data_process(D.get_data('BTC'))
data.indic()
#data.ma_tranf()

plt.plot(data.df['Close price'])
plt.show()

#prep data
data = D.data_input_LSTM(data.df, #data pd.df
                         15, #predict t+4
                         6, #timestep
                         'Close price',#output
                         is_reg=True) 
X, y = data.split_sequences()

#split train, test
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#scale output and input
X_train, scaler_x = D.scaling_X(X_train)
X_test, _ = D.scaling_X(X_test, scaler_x)

Y_train, scaler_y = D.scaling_Y(Y_train)
Y_test = scaler_y.transform(Y_test.reshape(-1, 1))


#création model
model = L.lstm_reg_model(data.timestep, X_train.shape[2])
# fit network
history = model.fit(X_train, Y_train, epochs=100, batch_size=72, validation_data=(X_test, Y_test), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Make predictions on the test data
y_pred = model.predict(X_test)

#unscale_data
y_pred = scaler_y.inverse_transform(y_pred)
Y_test = scaler_y.inverse_transform(Y_test)
Y_train = scaler_y.inverse_transform(Y_train)

#creation colonne pred
col_pred = np.concatenate((Y_train, y_pred), axis=0)
df_pred = data.df.iloc[(data.timestep - 1):-1]
df_pred = df_pred.assign(test_pred=col_pred)
df_pred['Close price'] = df_pred['Close price'].shift(data.shift)
df_pred['test_pred'] = df_pred['test_pred'].shift(data.shift)

#trend reward 0.01 coef + malus
df_plot = df_pred.iloc[-len(y_pred):]
plt.plot(df_plot['test_pred'], color = 'blue', label = 'pred')
plt.plot(df_plot['Close price'], color = 'red', label = 'test')
plt.legend()
plt.show()
