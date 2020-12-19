import numpy as np 
import sys

test_dir = "C:\\Users\\yeyun\\Google Drive\\university\\2020 fall\\CS 145\\covid-19-case-prediction-project\\data\\"
test_file = "sept_final.csv"
predict_dir = "C:\\Users\\yeyun\\Google Drive\\university\\2020 fall\\CS 145\\covid-19-case-prediction-project\\"


def mean_absolute_percentage_error(y_true, y_pred): 
    """
    Calculate MAPE as sklearn.metrics seems not updated to have it
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():
    for lag in [7,9,10,12,15,17,20]:
        for ma in [2,4,6]:
            predict_file =  'arima_reference%d%d.csv' % (lag, ma)
            test_data = np.genfromtxt(test_dir + test_file, dtype=int, delimiter=',', names=True)
            try:
                predict_data = np.genfromtxt(predict_dir + predict_file, dtype=int, delimiter=',', names=True)
            except:
                continue
            confirmed_mape = mean_absolute_percentage_error(test_data['Confirmed'], predict_data['Confirmed'])
            death_mape = mean_absolute_percentage_error(test_data['Deaths'], predict_data['Deaths'])
            mape = (confirmed_mape + death_mape) / 2
            print('MAPE  is: %f for lag %d and ma %d' % (mape,lag,ma))

if __name__ == '__main__':
    main()