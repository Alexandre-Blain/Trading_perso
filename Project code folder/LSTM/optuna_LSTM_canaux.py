from Extract_data.extract import Finance_Data
import LSTM.data_processing_LSTM as D
import LSTM.LSTM_model as L
from sklearn.model_selection import train_test_split
from sklearn import metrics
import optuna

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



# start optuna
def objective(trial):
    list_train, list_test = train_test_split(klines, test_size=0.2, shuffle=True)
    ep = trial.suggest_int("ep", 10, 150)

    params = {'learning_rate': trial.suggest_float("learning_rate", 0.0001, 0.2),
              #'activation': trial.suggest_categorical("activation", ['sigmoid', 'relu', 'tanh', 'elu', 'gelu']),
              #'dense_units': trial.suggest_int("dense_units", 10, 150),
              'lstm_units': trial.suggest_int("lstm_units", 20, 150),
              'n_layer': trial.suggest_int("n_layer", 1, 10),
              'dropout_rate': trial.suggest_float("dropout_rate", 0.01, 0.2)}

    model = L.lstm_reg_model(trial,
                             10,
                            list_train[0][0].shape[2],
                            #dense_units=params['dense_units'],
                            lstm_units=params['lstm_units'],
                            dropout_rate=params['dropout_rate'],
                            learning_rate=params['learning_rate'],
                            #activation=params['activation'],
                            n_layer=params['n_layer'],
                            loss=L.canal_high_loss())

    for i in list_train:
        model.fit(i[0], i[1], epochs=ep, batch_size=36, verbose=0, shuffle=False)

    list_mse = []
    for j in list_test:
        y_pred = model.predict(j[0])
        mse = metrics.mean_squared_error(j[1], y_pred)
        list_mse.append(mse)
        err = sum(list_mse) / len(list_mse)

    return err

study = optuna.create_study(direction="minimize",
                            study_name='Optuna_LSTM_canal',
                            storage='sqlite:///C:\\Users\\alexa\\OneDrive\\Bureau\\5A_Project\\study_optuna_canal.db',
                            load_if_exists = True)

study.optimize(objective, n_trials=40, show_progress_bar=True)
