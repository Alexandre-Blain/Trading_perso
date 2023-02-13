from binance import Client
import numpy as np
import os
import pandas as pd

class Finance_Data:
    
    def __init__(self):
        
        with open("Key.txt") as k:
            key = k.readline()
            key = key.rstrip('\n').split(' ')
        api_key=key[0]
        api_secret=key[1]
        
        self.client = Client(api_key, api_secret)
        pass
    


    def extract(self, liste, is_intervalle, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        name = ''
        j = 0
        if is_intervalle:

            for i in liste:

                if name != i[0]:
                    j = 0
                j += 1

                name = i[0]
                date_min = i[1]
                date_max = i[2]

                data = self.client.get_historical_klines(name + 'USDT', Client.KLINE_INTERVAL_30MINUTE, date_min, date_max)
                df = pd.DataFrame(np.array(data)[:, 0:11], index=np.arange(len(data)),
                                  columns=['Open time', 'Open price', 'High price', 'Low price', 'Close price',
                                           'Volume', 'Close time', 'Quote asset volume', 'Number of trades',
                                           'Taker buy base asset volume', 'Taker buy quote asset volume'])

                df.to_csv(f'{folder_name}/{name}_{j}.csv', index=False)
        else:
            for i in liste:
                name = i[0]
                date_min = i[1]

                data = self.client.get_historical_klines(name + 'USDT', Client.KLINE_INTERVAL_4HOUR, date_min)
                df = pd.DataFrame(np.array(data)[:, 0:11], index=np.arange(len(data)),
                                  columns=['index', 'Open price', 'High price', 'Low price', 'Close price', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume'])

                df.to_csv(f'{folder_name}/{name}_{j}.csv', index=False)

    def load_csv_folder(self, folder_name):
        klines = []
        
        for file_name in os.listdir(folder_name):
            file_path = os.path.join(folder_name, file_name)
            klines.append(pd.read_csv(file_path))

        return klines
