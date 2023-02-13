import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.signal import find_peaks

class Droite:
    # attribute:
    # X_reg ; Y_reg ; model ; X_pred ; Y_pred ; MSE ; R2
    
    def __init__(self, X_reg, Y):
        self.X_reg=X_reg
        self.Y_reg=Y
        self.reg_lin()
        pass
    
    def reg_lin(self, X_pred=np.array(1)):
        if X_pred.shape == ():
            X_pred = np.linspace(min(self.X_reg),max(self.X_reg)+26000000,len(self.X_reg)).reshape(-1,1)
        self.model = LinearRegression()
        self.model.fit(self.X_reg,self.Y_reg) 
        self.Y_pred=self.model.predict(X_pred)
        self.MSE=mean_squared_error(self.Y_reg, self.Y_pred[:len(self.Y_reg)])
        self.R2=self.model.score(self.X_reg,self.Y_reg)
        self.X_pred=X_pred
        pass

class Canal:
    #attribute:
    # droite_max ; droite_min ; pente(1 croissant, -1 décroissant) ; coef_moy (coef directeur moyen); forme; intervalle
    def __init__(self, droite_max, droite_min, a, b):
        self.droite_max = droite_max
        self.droite_min = droite_min
        self.intervalle = [a, b]
        self.coef_moy = (self.droite_max.model.coef_ + self.droite_min.model.coef_)/2
        
        self.Pente()
        self.Forme()
        pass
    
    def Pente(self):
        if self.coef_moy>0:
            self.pente=1
        else:
            self.pente=-1
    
    def Forme(self, conf=0.075):
        dif1 = abs(self.droite_min.model.predict(self.droite_max.X_reg[-1].reshape(-1, 1)) -self.droite_max.model.predict(self.droite_max.X_reg[-1].reshape(-1, 1)))
        dif2 = abs(self.droite_min.model.predict(self.droite_max.X_reg[-2].reshape(-1, 1)) -self.droite_max.model.predict(self.droite_max.X_reg[-2].reshape(-1, 1)))
        
        int_conf=conf*dif2
        dif= dif1 - dif2
        
        if dif<int_conf:#parallèle
            self.forme=0
        elif dif<0:#fermé
            self.forme=-1
        else:#ouvert
            self.forme=1

class Cerveau:
    trade=[]
    len_moy_mini = []
    len_moy_max = []
    #attribute:
    # len ; activity ; canaux (liste d'objets Canal), trade (liste d'objets trade)
    def __init__(self, df):
        self.activity = 0
        df=df.apply(pd.to_numeric)
        
        #spécifique MACD
        df.ta.macd(close = 'close', fast = 12, slow = 26, signal = 9, append = True)
        df.ta.mfi(high = 'High price', low = 'Low price', close = 'Close price', volume = 'Volume', length=14, drift=1, append=True)
        
        self.df = df
        
        self.process_df()
        pass
    
    def update(self, maj):#comparer r2 update vs old
        sub_df = pd.concat([self.df.iloc[len(self.df)-40:], maj], axis=0)
        sub_df.ta.macd(close = 'close', fast = 12, slow = 26, signal = 9, append = True)
        sub_df.ta.mfi(high = 'High price', low = 'Low price', close = 'Close price', volume = 'Volume', length=14, drift=1, append=True)
        self.df = pd.concat([self.df.iloc[1:], sub_df.iloc[-1]], axis=0)
        self.process_df()
    
    def process_df(self):
        #cherche indice ou il y a changement de signe puis création des intervalles de ces changements
        s = np.array(np.sign(self.df['MACDh_12_26_9']).diff().ne(0))
        s = np.where(s == True)[0]

        intervalle = []
        for i in range (len(s)-1):
            intervalle.append([s[i],s[i+1]])
        intervalle.append([s[-1],len(self.df)])
        
        self.canaux=[]
        #cherche les max/min locaux sur toutes la courbes
        peaks_maxi, _ = find_peaks(self.df['High price'], distance= 1)
        peaks_mini, _ = find_peaks(-self.df['Low price'], distance= 1)

        #application de la regression sur chaque intervalle
        for i in range (len(intervalle)):
            a = intervalle[i][0]
            b = intervalle[i][1]


            #condition d'avoir au moins 3 peaks max sur l'intervalle pour faire la regression
            c_maxi = np.where((peaks_maxi < b) & (peaks_maxi > a))[0]
            c_mini = np.where((peaks_mini < b) & (peaks_mini > a))[0]

            if ((len(c_maxi) >= 2) & (len(c_mini) >= 3)):#double cross macd (7-8 bougie bonne métrique)
                self.len_moy_max.append(c_maxi)
                self.len_moy_mini.append(c_mini)

                if i == (len(intervalle)-1):#dernier intervalle(intervalle actuelle dans un canal)
                    self.activity=1

                #essayer plusieurs reg pour garder la meilleurs
                X_maxi = np.array(self.df.iloc[peaks_maxi[c_maxi], self.df.columns.get_loc('index')]).reshape(-1, 1)
                Y_maxi = np.array(self.df.iloc[peaks_maxi[c_maxi], self.df.columns.get_loc('High price')]).reshape(-1, 1)
                X_mini = np.array(self.df.iloc[peaks_mini[c_mini], self.df.columns.get_loc('index')]).reshape(-1, 1)
                Y_mini = np.array(self.df.iloc[peaks_mini[c_mini], self.df.columns.get_loc('Low price')]).reshape(-1, 1)

                c = Canal(Droite(X_maxi,Y_maxi), Droite(X_mini,Y_mini), a, b)
                self.canaux.append(c)
        
        pass
    
    def trading_order(self):
        if self.activity == 1:#canaux en cours de construction
            if self.canaux[-1].pente==1:#ascendant
                if self.canaux[-1].droite_min.model.predict(Cerveau.last_df['Close time'].iloc[-1].reshape(-1, 1)) >= Cerveau.last_df['Close price'].iloc[-1]:
                    l=1
                    Cerveau.trade.append(Trade(Cerveau.last_df, 1, l))
                    return Cerveau.trade[-1]
                else:
                    return print('Long : point de départ pas encore atteint')
            else:#descendant
                if self.canaux[-1].droite_max.model.predict(Cerveau.last_df['Close time'].iloc[-1].reshape(-1, 1)) <= Cerveau.last_df['Close price'].iloc[-1]:
                    l=1 #calcul effet levier
                    Cerveau.trade.append(Trade(Cerveau.last_df, -1, l))
                    return Cerveau.trade[-1]
                    
                else:
                    return print('Short : point de départ pas encore atteint')
        else:
            print('Pas de canaux en cours')
            pass
        
        #si recroisemment MACD quitter trade en cours
        #check hit des stops loss --> actions a effectuer :
        # - virer trade de liste des trades
        # - potentiel bet opposé

class Trade(Cerveau):
    #Attribute
    #bet (1 long/ -1 short) ; start_time ; start_price ; max ; levier ; sl (stop loss)
    def __init__(self, df, bet, l):#sub le df?
        self.bet = bet
        self.start_time = df['Close time'].iloc[-1]
        self.start_price = df['Close price'].iloc[-1]
        self.levier = l
        self.max = df['Close price'].iloc[-1]
        self.set_max(df)
        
    
    def set_max(self, df):
        df = df[df['Close time'] >= self.start_time]
        if self.bet == 1:
            self.max = max(df['High price']) if self.max < max(df['High price']) else self.max
            self.set_stop_loss()
        else:
            self.max = min(df['Low price']) if self.max > max(df['High price']) else self.max
            self.set_stop_loss()
        pass
            
    def set_stop_loss(self, conf= 0.05):
        if self.bet == 1:
            self.sl = self.max*(1-conf)
        else:
            self.sl = self.max*(1+conf)
        pass