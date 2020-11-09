import numpy as np 

test_dir = "C:\\Users\\yeyun\\Google Drive\\university\\2020 fall\\CS 145\\covid-19-case-prediction-project\\data\\"
test_file = "sept_final.csv"
predict_dir = "C:\\Users\\yeyun\\Google Drive\\university\\2020 fall\\CS 145\\covid-19-case-prediction-project\\"
predict_file = "ar_submission.csv"

def mean_absolute_percentage_error(y_true, y_pred): 
    """
    Calculate MAPE as sklearn.metrics seems not updated to have it
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():
    test_data = np.genfromtxt(test_dir + test_file, dtype=int, delimiter=',', names=True)
    predict_data = np.genfromtxt(predict_dir + predict_file, dtype=int, delimiter=',', names=True)
    confirmed_mape = mean_absolute_percentage_error(test_data['Confirmed'], predict_data['Confirmed'])
    death_mape = mean_absolute_percentage_error(test_data['Deaths'], predict_data['Deaths'])
    mape = (confirmed_mape + death_mape) / 2
    print('MAPE: %f' % mape)

if __name__ == '__main__':
    main()