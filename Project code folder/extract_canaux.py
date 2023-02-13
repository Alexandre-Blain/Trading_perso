import Live_channel.channel_class as cc
import LSTM.data_processing_LSTM as dp
import Extract_data.extract as extract

cry = ['BTC','ETH','SOL','MATIC','XRP','ATOM','CHZ','APE','NEAR', 'DOGE', 'BNB', 'ADA']
liste = []
save = extract.Finance_Data()

for crypto in cry:
    print(crypto)

    data = dp.data_process(dp.get_data(crypto, '1 Avr, 2021'), is_binance=True)
    
    df = data.df.reset_index()
    brain = cc.Cerveau(df)
    
    
    for i in brain.canaux:
        sub_list = []
        low = i.intervalle[0]
        hight = i.intervalle[1]
        date_min = df['index'].iloc[low-1]
        date_max = df['index'].iloc[hight-1] if len(df) < (hight+30) else df['index'].iloc[hight+30]
        sub_list.extend([crypto, str(date_min), str(date_max)])
        liste.append(sub_list)

save.extract(liste, is_intervalle=True, folder_name='Canaux_30m')