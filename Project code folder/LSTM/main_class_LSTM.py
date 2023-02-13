import data_processing_LSTM as D
import LSTM_model as L
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics

labels = ['Decrease of more than 5%', 'Decrease of 0.5-5%', 'No change +-0.5%', 'Increase of 0.5-5%', 'Increase of more than 5%']

weights = np.array([3, 3, 1, 3, 3])
#get data from binance
data = D.data_process(D.get_data('SOL'))
data.categorical_column()
data.indic()
print(data.df['f_trend_close_category'].value_counts())
#pred data
data = D.data_input_LSTM(data.df, #data pd.df
                         4, #predict t+4
                         5, #timestep
                         'f_trend_close_category',#output
                         is_reg=False) 
X, y = data.split_sequences()

#y_dum = pd.get_dummies(y).astype('float32').values
y_dum = pd.get_dummies(y).astype('float32')
y_dum = y_dum.reindex(labels, axis = 1)
y_dum = y_dum.astype('float32').values
#split test train
X_train, X_test, Y_train, Y_test = train_test_split(X, y_dum, test_size=0.2, shuffle=False)

#scale output and input
X_train, scaler_x = D.scaling_X(X_train)
X_test, _ = D.scaling_X(X_test, scaler_x)



#création model
model = L.lstm_class_model(Y_train.shape[1], data.timestep, X_train.shape[2], loss = L.weighted_categorical_crossentropy_vec(weights))
# fit network
history = model.fit(X_train, Y_train, epochs=60, batch_size=72, validation_data=(X_test, Y_test), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Make predictions on the test data
y_pred = model.predict(X_test)

print(model.evaluate(X_train, Y_train))
err = metrics.matthews_corrcoef(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(f'err = {err}')

confusion_matrix = metrics.confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1))
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,  display_labels = labels)
cm_display.plot()
plt.show()
