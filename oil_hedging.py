import pandas as pd
import numpy as np
import pickle
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import keras

def learning(data_path):
    '''

    :param data_path:
    :return:
    '''
    f_data = pd.read_table(data_path, sep=',')
    f_data = f_data.dropna()
    model = Sequential()
    model.add(Dense(24, input_dim=10, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))

    # model.add(Dense(24, input_dim=8, activation='sigmoid'))
    # model.add(Dense(16, activation='sigmoid'))
    # model.add(Dense(8, activation='sigmoid'))
    # model.add(Dense(1))


    model.compile(loss='mean_squared_error', optimizer='adam')
    validation_size = 0.30
    seed = 0
    # X = f_data[['pct5_macd_5', 'pct5_macdsignal_5', 'pct5_macdhist_5', 'd5_rsi', 'pct5_vol', 'pct5_macd_20', 'pct5_macdsignal_20', 'pct5_macdhist_20', 'b_upper','b_lower']].as_matrix()
    X = f_data[['pct5_macd_5', 'pct5_macdsignal_5', 'pct5_macdhist_5', 'd5_rsi', 'pct5_vol', 'pct5_macd_20',
                'pct5_macdsignal_20', 'pct5_macdhist_20', 'b_upper', 'b_lower']].as_matrix()
    Y = f_data['pct5_sma_5'].as_matrix()
    # X = f_data[['d_macd', 'd_macdsignal', 'd_macdhist', 'd_rsi']]
    # Y = f_data['d_price']


    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    # print(X_train)
    tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(X_train, Y_train, epochs=2000, batch_size=50, callbacks=[tb_hist])
    Y_prediction = model.predict(X_validation).flatten()
    for pre, val in zip(Y_prediction, Y_validation):
        print('predicted price= {:.3f}, real price = {:.3f}, diff ={:.3f}'.format(pre, val, pre-val))

    # print(X)
    # print(Y_prediction)
    # Y_prediction.plot()
    # np.random.seed(19680801)
    #
    # fig, ax = plt.subplots()
    #
    # # ax.plot(np.random.rand(20), '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
    # ax.plot(Y_prediction, '-o')
    #
    # # ax.plot([Y_prediction], '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
    # # ax.plot([Y_validation], '-o', ms=20, lw=2, alpha=0.7, mfc='orange')
    #
    # ax.grid()
    #
    # # position bottom right
    # fig.text(0.95, 0.05, 'Property of MPL',
    #          fontsize=50, color='gray',
    #          ha='right', va='bottom', alpha=0.5)
    #
    # plt.show()
    # ax.plot(x=np.ndarray([1,2,3]), y = np.ndarray([1,2,3]), kind='scatter')
    # plt.show()


learning('c:\\fine_oil_price.txt')