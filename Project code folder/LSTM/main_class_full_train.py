import data_processing_LSTM as D
import LSTM_model as L
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#get data from binance
data = D.data_process(D.get_data())
data.categorical_column()
data.indic()

#pred data
data = D.data_input_LSTM(data.df, #data pd.df
                         4, #predict t+4
                         5, #timestep
                         'f_trend_close_category',#output
                         is_reg=False) 
X, y = data.split_sequences()

y = pd.get_dummies(y).astype('float32').values
#split test train
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#scale output and input
X_train, scaler_x = D.scaling_X(X_train)
X_test, _ = D.scaling_X(X_test, scaler_x)



#création model
model = L.lstm_class_model(Y_train.shape[1], data.timestep, X_train.shape[2])
# fit network
history = model.fit(X_train, Y_train, epochs=60, batch_size=72, validation_data=(X_test, Y_test), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Make predictions on the test data
y_pred = model.predict(X_test)

model.evaluate(X_test, Y_test)
