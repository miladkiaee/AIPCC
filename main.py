import funcs

import pandas as pd

import time
import seaborn as sns
#import matplotlib
#matplotlib.use('TkAgg')
import pylab
import datetime
import numpy as np

ls_time = "20000101"

coins = ['btc', 'ada', 'eth', 'grc', 'xrp', 'xem']

for coin in coins:
    coin_name_str = funcs.coin_name(coin);
    print("Coin is "),
    print(coin),
    print(" Name is "),
    print(coin_name_str)

    coin_info = pd.read_html("https://coinmarketcap.com/currencies/" + coin_name_str + "/historical-data/?start=" + ls_time + "&end=" + time.strftime("%Y%m%d"))[0]

    print(coin_info.head())
    print(coin_info.dtypes)

    # convert the date string to the correct date format
    coin_info = coin_info.assign(Date=pd.to_datetime(coin_info['Date']))

    # convert to int

    coin_info.replace({'-': '0'}, regex=True)
#    coin_info.drop('\n', axis=1, inplace=True)

#    coin_info['Volume'] = coin_info['Volume'].str.replace("-", "0")

    #coin_info['Volume'] = coin_info['Volume'].astype('int64')

    coin_info.to_csv(coin_name_str);

    ###################################################3
    coin_ts = coin_info[['Date', 'Close']]

    coin_ts.set_index('Date', inplace=True)
    pylab.ylabel("USD")
    pylab.title(coin)

    print(coin_ts.plot())

    pylab.show()



