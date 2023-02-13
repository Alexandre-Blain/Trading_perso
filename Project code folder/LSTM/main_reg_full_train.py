import LSTM.data_processing_LSTM as D
import LSTM.LSTM_model as L
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics



u = 0
scaler_dict = {}
data_dict = {}
#get data from binance
crypto = ['BTC','ETH','SOL','MATIC','XRP','ATOM','CHZ','APE','NEAR', 'DOGE']
for i in crypto:
    u+=1
    #get data from binance
    data = D.data_process(D.get_data(i))
    data.indic()
    
    #pred data
    data = D.data_input_LSTM(data.df, #data pd.df
                             5, #predict t+4
                             6, #timestep
                             'Close price',#output
                             is_reg=True) 
    X, y = data.split_sequences()


    #split test train
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    X_train, scaler_x = D.scaling_X(X_train)
    X_test, _ = D.scaling_X(X_test, scaler_x)
    
    Y_train, scaler_y = D.scaling_Y(Y_train)
    Y_test = scaler_y.transform(Y_test.reshape(-1, 1))
        
    scaler_dict[i] = [scaler_x, scaler_y]
    data_dict[i] = [X_test, Y_test]
    
   
    if u==1:
        model = L.lstm_reg_model(data.timestep, X_train.shape[2])

    # fit network
    history = model.fit(X_train, Y_train, epochs=10, batch_size=36, validation_data=(X_test, Y_test), verbose=2, shuffle=False)
    # plot history
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()

for i in crypto:
    y_pred = model.predict(data_dict[i][0])
    
    rmse = np.sqrt(metrics.mean_squared_error(data_dict[i][1], y_pred))
    
    y_pred = scaler_dict[i][1].inverse_transform(y_pred)
    Y_test = scaler_dict[i][1].inverse_transform(data_dict[i][1])
    

    
    print(f'Pour {i}, RMSE = {rmse}')
    plt.plot(y_pred, color = 'blue', label = 'prediction')
    plt.plot(Y_test, color = 'red', label = 'Test')
    plt.title(f'Pour {i}, RMSE = {rmse}')
    plt.legend()
    plt.show()
