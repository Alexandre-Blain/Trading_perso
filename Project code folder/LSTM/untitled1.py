import data_processing_LSTM as D
import LSTM_model as L
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import optuna

def objective(trial):

    fat = trial.suggest_float("fat", 0.1, 0.99)
    med = trial.suggest_float("med", 0.1, 0.99)
    low = trial.suggest_float("low", 0.9, 1.5)
    opo = trial.suggest_float("opo", 1.0, 15.0)
    same = trial.suggest_float("same", 0.5, 1.2)
    opo_less = trial.suggest_float("opo_less", 1.0, 15.0)
    little = trial.suggest_float("opo_less", 1.0, 15.0)
    
    weights = np.array([[fat, same, opo_less, opo_less, opo],
                    [same, med, opo_less, opo_less, opo],
                    [opo, little, low, little, opo],
                    [opo, opo_less, opo_less, med, same],
                    [opo, opo_less, opo_less, same, fat]])
    u = 0
    #get data from binance
    crypto = ['BTC','ETH','SOL','MATIC','XRP','ATOM','CHZ','APE','NEAR', 'DOGE']
    for i in crypto:
        u+=1
        #get data from binance
        data = D.data_process(D.get_data(i))
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
        
        if u==1:
            params={'learning_rate': 0.0008857250003024605, 'activation': 'tanh', 'dense_units': 48, 'lstm_units': 22, 'n_layer': 2, 'dropout_rate': 0.03251418561585747}
            
            model = L.lstm_class_model(Y_train.shape[1], 
                                           data.timestep, 
                                           X_train.shape[2], 
                                           dense_units=params['dense_units'], 
                                           lstm_units=params['lstm_units'], 
                                           dropout_rate=params['dropout_rate'], 
                                           learning_rate=params['learning_rate'], 
                                           activation = params['activation'],
                                           n_layer = params['n_layer'],
                                           loss = L.weighted_categorical_crossentropy(weights))
    
            X_stack = X_test
            Y_stack = Y_test
        else :
            X_stack = np.vstack((X_stack, X_test))
            Y_stack = np.vstack((Y_stack, Y_test))
        
        # fit network
        model.fit(X_train, Y_train, epochs=10, batch_size=36, validation_data=(X_test, Y_test), verbose=2, shuffle=False)
    
    
    # Make predictions on the test data
    y_pred = model.predict(X_stack)
    err = metrics.matthews_corrcoef(Y_stack.argmax(axis=1), y_pred.argmax(axis=1))
    
    return err

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=60)

study.best_trial
