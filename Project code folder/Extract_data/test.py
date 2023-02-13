# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 20:59:53 2023

@author: alexa
"""

import extract

crypto = ['ETH', 'BTC', 'DOGE']
#date = ['1 Jan, 2023','10 Jan, 2022'] #format date_min
date = [['1 Jan, 2023','10 Jan, 2023'], ['1 Jan, 2022','10 Jan, 2022']]#format intervalles
test = extract.Finance_Data()
test.extract(crypto, date, False, 'canal_max')