import pandas as pd
import ccxt

exchange = ccxt.okx()
exchange.https_proxy = 'http://127.0.0.1:7897/'

symbol = 'BTC/USDT'
time_interval = '5m'

ohlcv = exchange.fetch_ohlcv(symbol, time_interval, limit=100)
print(ohlcv)