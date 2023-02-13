import pandas as pd
import pandas_ta as ta
import numpy as np
from binance import Client
from sklearn.preprocessing import MinMaxScaler

def get_data(crypto):
    with open("Key.txt") as k:
        key = k.readline()
        key = key.rstrip('\n').split(' ')
    api_key=key[0]
    api_secret=key[1]
    
    client = Client(api_key, api_secret)
    klines = client.get_historical_klines(crypto+'USDT', Client.KLINE_INTERVAL_4HOUR)
    return(klines)

class data_process():
    
    def __init__(self, klines):
        #création du df a partir des données Binance
        df_klines = pd.DataFrame(np.array(klines)[:,0:11], index = np.arange(len(klines)), columns = ['Open time', 'Open price', 'High price', 'Low price', 'Close price', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume'])
        df_klines = df_klines.apply(pd.to_numeric)
        df_klines['Open time'] = pd.to_datetime(df_klines['Open time'], unit='ms')
        df_klines = df_klines.set_index('Open time')
        df_klines = df_klines.drop('Close time', axis = 1)
        
        df_klines = df_klines.assign(f_trend_close = df_klines['Close price'].pct_change())
        self.df = df_klines.dropna()
        pass
    
    def indic(self):
        #ajour des incic au df
        self.df.ta.macd(close='Close price', fast=12, slow=26, signal=9, append=True)
        self.df.ta.rsi(close = 'Close price', length = 14, scalar = 100, drift = 1, offset = 0, append = True)
        self.df.ta.mfi(high = 'High price', low = 'Low price', close = 'Close price', volume = 'Volume', length=14, drift=1, append=True)
        self.df['ma8'] = ta.ma("sma", self.df['Close price'], length=4)
        self.df.dropna(inplace = True)
        pass
    
    def log_tranf(self):
        self.df['Close price'] = np.log(self.df['Close price'])
        
    def ma_tranf(self):
        self.df['Close price'] = self.df['Close price'] - self.df['ma8']
    
    def categorical_column(self):
        bins = [-1, -0.07, -0.005, 0.005, 0.07, 1]
        labels = ['Decrease of more than 5%', 'Decrease of 0.5-5%', 'No change +-0.5%', 'Increase of 0.5-5%', 'Increase of more than 5%']
        self.df['f_trend_close_category'] = pd.cut(self.df['f_trend_close'], bins=bins, labels=labels, include_lowest=True)
        pass
    
    def create_increment_column(self):
        increment_column = [0]
        previous_sign = self.df.iloc[0]['f_trend_close']
        increment = 0
        for i in range(1,len(self.df)):
            # Get the sign of the current element
            current_sign = self.df.iloc[i]['f_trend_close']
            
            # If the sign is the same as the previous element, increment the counter
            if (current_sign > 0 and previous_sign > 0) or (current_sign < 0 and previous_sign < 0):
                increment += 1
            # If the sign is different from the previous element, reset the counter
            else:
                increment = 0 if increment <= 1 else increment - 2
            
            # Append the counter to the increment column
            increment_column.append(increment)
            
            # Update the previous sign
            previous_sign = current_sign
        
            # Add the increment column to the DataFrame
            self.df['increment'] = increment_column
            pass

class data_input_LSTM():
    #shift -> pred t+shift ; timestep garde en mémoire timestep 
    def __init__(self, df, shift, timestep, output, is_reg):
        self.df = df
        self.shift = - shift
        self.timestep = timestep
        self.output = output
        self.is_reg = is_reg
        
        self.output_last()
        if is_reg:
            self.df = self.df.assign(f_output = self.df[output])
        self.df[output] = self.df[output].shift(shift)
        self.df.dropna(inplace = True)
        pass
    
    def output_last(self): 
        col = self.df.pop(self.output)
        # Insérer la colonne à la fin du DataFrame
        self.df.insert(len(self.df.columns), self.output, col)

    
    def split_sequences(self):
        sequences = self.df.values
        X, y = list(), list()
        for i in range(len(sequences)):
        # find the end of this pattern
            end_ix = i + self.timestep
        # check if we are beyond the dataset
            if end_ix > len(sequences)-1:
                break
        # gather input and output parts of the pattern (for predict x(t+1))
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
            X.append(seq_x)
            y.append(seq_y)
        
        return np.array(X), np.array(y)

def scaling_X(X, scaler = None):
    # Normalize the data
    num_samples, num_timesteps, num_features = X.shape
    two_d_shape = X.reshape(num_samples, num_timesteps * num_features)
    if scaler == None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        two_d_shape = scaler.fit_transform(two_d_shape)
    else:
        two_d_shape = scaler.transform(two_d_shape)
    X = two_d_shape.reshape(num_samples, num_timesteps, num_features)
    return X, scaler

def scaling_Y(Y):
    scaler = MinMaxScaler(feature_range=(0, 1))
    Y = scaler.fit_transform(Y.reshape(-1, 1))
    return Y, scaler

        
    