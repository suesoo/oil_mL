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

    # model.add(Dense(24, input_dim=10, activation='sigmoid'))
    # model.add(Dense(16, activation='sigmoid'))
    # model.add(Dense(8, activation='sigmoid'))
    # model.add(Dense(1))


    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    valid_size = 0.20
    test_size = 0.20
    seed = 0
    # X = f_data[['pct5_macd_5', 'pct5_macdsignal_5', 'pct5_macdhist_5', 'd5_rsi', 'pct5_vol', 'pct5_macd_20', 'pct5_macdsignal_20', 'pct5_macdhist_20', 'b_upper','b_lower']].as_matrix()
    X = f_data[['pct5_macd_5', 'pct5_macdsignal_5', 'pct5_macdhist_5', 'd5_rsi', 'pct5_vol', 'pct5_macd_20',
                'pct5_macdsignal_20', 'pct5_macdhist_20','b_upper', 'b_lower']].as_matrix()

    # X = f_data[['pct5_macd_5', 'pct5_macdsignal_5', 'pct5_macdhist_5', 'd5_rsi', 'pct5_vol', 'pct5_macd_20',
    #             'pct5_macdsignal_20', 'pct5_macdhist_20', 'b_upper', 'b_lower','pct5_dollar_idx']].as_matrix()
    # X = f_data[['pct5_macd_5', 'pct5_macdsignal_5', 'pct5_macdhist_5', 'd5_rsi', 'pct5_vol', 'pct5_macd_20',
    #             'pct5_macdsignal_20', 'pct5_macdhist_20']].as_matrix()
    Y = f_data['pct5_sma_5'].as_matrix()
    # X = f_data[['d_macd', 'd_macdsignal', 'd_macdhist', 'd_rsi']]
    # Y = f_data['d_price']

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
    X_train, X_valid, Y_train, Y_valid = model_selection.train_test_split(X_train, Y_train, test_size=valid_size, random_state=seed)

    tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(patience=30)  # 조기종료 콜백함수 정의
    hist = model.fit(X_train, Y_train, epochs=2000, batch_size=25, validation_data=(X_valid, Y_valid), callbacks=[tb_hist, early_stopping])
    # hist = model.fit(X_train, Y_train, epochs=2000, batch_size=50, validation_data=(X_valid, Y_valid),
    #                  callbacks=[tb_hist])
    # 5. 모델 학습 과정 표시하기

    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')

    plt.show()

    loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=50)
    print('')
    print('loss_and_metrics : ' + str(loss_and_metrics))

    # 6. 모델 저장하기
    from keras.models import load_model
    model.save('c:\\ml_test\\oil_weekly_model.h5')
    Y_prediction = model.predict(X_test).flatten()
    correct_sign=0
    total_sample = 0
    for pre, val in zip(Y_prediction, Y_test):
        total_sample += 1
        if pre * val > 0:
            correct_sign += 1
        print('predicted price= {:.3f}, real price = {:.3f}, diff ={:.3f}'.format(pre, val, pre-val))
    print('hit ratio = ',correct_sign/total_sample)
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


learning('c:\\ml_test\\fine_oil_price.txt')