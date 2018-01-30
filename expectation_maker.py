# 0. 사용할 패키지 불러오기
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import argmax

# 1. 데이터 불러오기
data_path ='c:\\ml_test\\fine_oil_price.txt'
f_data = pd.read_table(data_path, sep=',')

# print(f_data)

f_data = f_data.dropna()
xhat = f_data[['pct5_macd_5', 'pct5_macdsignal_5', 'pct5_macdhist_5', 'd5_rsi', 'pct5_vol', 'pct5_macd_20',
            'pct5_macdsignal_20', 'pct5_macdhist_20', 'b_upper', 'b_lower']].as_matrix()

# 2. 모델 불러오기
from keras.models import load_model
model = load_model('c:\\ml_test\\oil_weekly_model.h5')

# 3. 모델 사용하기
yhat = model.predict(xhat).flatten()
f_data['y_fcst'] = yhat
f_data['model_price'] = f_data['sma_5'] * (1+yhat/100)
f_data['real_price'] = f_data['sma_5'] * (1+f_data['pct5_sma_5']/100)
f_data['model_price']
# for i in range(5):
#     print('True : ' + str(argmax(y_test[xhat_idx[i]])) + ', Predict : ' + str(yhat[i]))

f_data.plot(x='date', y=['model_price','real_price'], kind='line')
plt.show()