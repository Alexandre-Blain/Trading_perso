from Extract_data.extract import Finance_Data
import LSTM.data_processing_LSTM as D
import LSTM.LSTM_model as L
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

loader = Finance_Data()
path = "C:\\Users\\alexa\\OneDrive\\Bureau\\Python trading\\Project code folder\\Canaux_30m"
klines = loader.load_csv_folder(path)

for i in range(len(klines)):
    klines[i] = klines[i].rename(columns={'Open time': 'index'})
    klines[i] = D.data_process(klines[i], is_binance=False)
    klines[i].indic()

    # generate output
    klines[i].df['max_price'] = klines[i].df['High price'].shift(-1)[::-1].rolling(10).max()[::-1]
    klines[i].df.dropna(inplace=True)
    # pred data
    data = D.data_input_LSTM(klines[i].df,  # data pd.df
                             0,  # predict t+4
                             10,  # timestep
                             'max_price',  # output
                             is_reg=True)
    X, y = data.split_sequences()

    X_scaled, scaler_x = D.scaling_X(X)
    Y_scaled, scaler_y = D.scaling_Y(y)

    klines[i] = [X_scaled, Y_scaled, scaler_x, scaler_y]

list_train, list_test = train_test_split(klines, test_size=0.2, shuffle=True)
model = L.lstm_reg_model(10, list_train[0][0].shape[2], loss = L.canal_high_loss)

for i in list_train:
    history = model.fit(i[0], i[1], epochs=10, batch_size=36, verbose=2, shuffle=False)

list_rmse = []
list_mse = []
list_mae = []
list_r = []

inc = 0
for j in list_test:
    inc += 1
    y_pred = model.predict(j[0])

    rmse = np.sqrt(metrics.mean_squared_error(j[1], y_pred))
    mse = metrics.mean_squared_error(j[1], y_pred)
    mae = metrics.mean_absolute_error(j[1], y_pred)
    r = metrics.r2_score(j[1], y_pred)
    list_rmse.append(rmse)
    list_mse.append(mse)
    list_mae.append(mae)
    list_r.append(r)

    y_pred = j[3].inverse_transform(y_pred)
    Y_test = j[3].inverse_transform(j[1])

    plt.plot(y_pred, color='blue', label='prediction')
    plt.plot(Y_test, color='red', label='Test')
    plt.title(f'Pour {inc}, RMSE = {rmse}')
    plt.legend()
    plt.show()

print(f'Avg_MSE {sum(list_mse) / len(list_mse)}, Avg_RMSE = {sum(list_rmse) / len(list_rmse)}, Avg_MaE = {sum(list_mae) / len(list_mae)}, Avg_R = {sum(list_r) / len(list_r)}')


# regression prédire le maximun/minimum sur les x prochaines bougies
# metric mse + interieur aux bornes
# un lstm inf un sup et si marge trop faible pas de trade ou trade opposer


