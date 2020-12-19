import csv
import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from reference import make_arima_model_and_predict

path = os.getcwd()
data_dir = path + "\\state_diff2\\"
lag = 7
ma = 4
output_name = 'arma_submission.csv'
state_names = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado',
             'Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho',
             'Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',
             'Maine','Maryland','Massachusetts','Michigan','Minnesota',
             'Mississippi','Missouri','Montana','Nebraska','Nevada',
             'New_Hampshire','New_Jersey','New_Mexico','New_York',
             'North_Carolina','North_Dakota','Ohio','Oklahoma','Oregon',
             'Pennsylvania','Rhode_Island','South_Carolina','South_Dakota',
             'Tennessee','Texas','Utah','Vermont','Virginia','Washington',
             'West_Virginia','Wisconsin','Wyoming']

tested = False


def init_random(shape):
    # init random distribution to small value
    return np.random.uniform(low=-0.1, high=0.1, size=shape)


def mean_absolute_percentage_error(y_true, y_pred): 
    """
    Calculate MAPE as sklearn.metrics seems not updated to have it
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calculate_from_diff(last, diffs):
    """
    Calculate values based on diffs
    """
    res = []
    for i in range(len(diffs)):
        if i == 0:
            cur = last + diffs[i]
        else:
            cur = res[-1] + diffs[i]
        res.append(int(round(cur)))
    return res


def data_process(state_name):
    """
    Read the csv file and convert it into numpy array
    
    Args:
        state_name: the state file to process
    
    Return:
        data: the complete numpy array of the csv file
        -1 if the file not exist
    """
    # read the corresponding csv file
    try:
        data_path = data_dir + state_name + ".csv"
        data = np.genfromtxt(data_path, dtype=int, delimiter=',', names=True)
    except:
        return -1

    # make the diffs start with 0 as it is originally empty for the first entry
    data['death_diff'][0] = 0
    data['confirmed_diff'][0] = 0
    return data


class arma_model():
    def __init__(self):
        self.train_values = list()
        self.lagged_values = list()
        self.error_values = list()
    
    def ar_fit(self,train, lag):
        """
        Find the parameters AR model by training data
        """
        train_length = len(train)
        self.train_values = train.tolist()
        # fit parameters
        x_values = list()
        for i in range(lag+1, train_length):
            past_value = train[i-lag:i].tolist()
            # past_value.append(1)
            x_values.append(past_value)
        y_values = train[lag+1:].tolist()
        self.ar = LinearRegression()
        self.ar.fit(x_values, y_values)
        self.lagged_values = x_values
        return self


    def ar_predict(self, lag,predict_length):
        # predict future values
        result = list()
        whole_values = self.train_values[:]
        train_length = len(whole_values)
        for i in range(predict_length):
            x_value = whole_values[-lag:]
            # x_value.append(1)
            predict_value = self.ar.predict([x_value])
            whole_values.append(predict_value[0])
            result.append(predict_value[0])

        return result


    def optimize(self, parameters):
        temp_ar, temp_ma = np.reshape(parameters, self.params.shape)
        predict_values = self.arma_predict(self.train_values, temp_ar, temp_ma)
        mape = mean_absolute_percentage_error(self.train_values, predict_values)
        return mape


    def arma_fit(self, train, lag, ma):
        """
        Find the parameters ARMA model by training data
        """
        self.train_values = train
        self.ar = init_random(lag)
        self.ma = init_random(ma)

        self.params = np.stack([self.ar, self.ma])
        res = minimize(fun=optimize, x0=self.params.flatten(), method='Nelder-Mead', options={'maxiter': 10000, 'disp': True})
        self.final_params = np.reshape(res.x, self.params.shape)
        return self


    def arma_predict(self, x_values , predict_length):
        """
        Predict future values based on fit arma model
        """
        result = list()
        # whole_values = self.train_values[:]
        # train_length = len(whole_values)
        # for i in range(predict_length):
        #     # ar predict 
        #     ar_x_value = whole_values[-lag:]
        #     # ar_x_value.append(1)
        #     ar_predict_value = self.ar.predict([ar_x_value])[0]
            
        #     # ma predict
        #     ma_x_value = ar_x_value
        #     for i in range(-ma,0):
        #         ma_x_value.append(self.error_values[i])
        #     ma_predict_value = self.ma.predict([ma_x_value])[0]
        #     arma_predict_value = ma_predict_value

        #     # append the final predicte result and used the difference between AR and ARMA as error term
        #     self.error_values.append(arma_predict_value - ar_predict_value)
        #     whole_values.append(arma_predict_value)
        #     result.append(arma_predict_value)
        noises = np.zeros((len(x_values), len(x_values[0])))
        result = np.zeros((len(x_values), len(x_values[0])))
        ar_length = self.ar.shape[1]
        ma_length = self.ma.shape[1]
        noises[:, 0:ma_length] = np.random.normal(size(predict_length, ma_length), scale=0.1)
        for i in range(self.lag, predict_length):
            ar_predict_value = np.sum(self.ar * np.flip(x_values[:, i - ar_length:i], axis=1), axis=1)
            ma_predict_value = np.sum(self.ma * np.flip(noises[:, i - ma_length:i], axis=1), axis=1)

            result[:, i] = ar_predict_value + ma_predict_value
            noises[:, i] = x[:, i] - result[:, t]

        return result

    def arma_predict_future(self, predict_length):
        result = list()
        whole_values = self.train_values[:]
        train_length = len(whole_values)
        for i in range(predict_length):
            # ar predict 
            ar_x_value = whole_values[-lag:]
            # ar_x_value.append(1)
            ar_predict_value = self.ar.predict([ar_x_value])[0]
            
            # ma predict
            ma_x_value = ar_x_value
            for i in range(-ma,0):
                ma_x_value.append(self.error_values[i])
            ma_predict_value = self.ma.predict([ma_x_value])[0]
            arma_predict_value = ma_predict_value

            # append the final predicte result and used the difference between AR and ARMA as error term
            self.error_values.append(arma_predict_value - ar_predict_value)
            whole_values.append(arma_predict_value)
            result.append(arma_predict_value)


def make_arma_prediction(data,lag,predict_length, ma):
    """
    Main function to create ARMA model and make prediction

    data: pandas dataframe that contains the training data
    lag: the number of lag days of AR model
    predict_length: the number of future values the model will predict
    ma: number of moving average model parameters
    """
    result = dict()

    confirmed_model = arma_model().arma_fit(data['confirmed_diff'], lag, ma)
    diff_confirmed_predict = confirmed_model.arma_predict(predict_length)
    death_model = arma_model().arma_fit(data['death_diff'], lag, ma)
    diff_death_predict = death_model.arma_predict(predict_length)

    result['confirmed'] = calculate_from_diff(data['Confirmed'][-1], diff_confirmed_predict)
    result['death'] = calculate_from_diff(data['Deaths'][-1], diff_death_predict)
    return result


def main():
    # run model
    submission_round = int(sys.argv[1])
    predict_length = 26 if submission_round == 1 else 21
    result = dict()
    for state in state_names:
        data = data_process(state)
        result[state] = make_arima_model_and_predict(data,lag,predict_length,ma)

    # generate submission dictionary
    submission = {'ForecastID':[], 'Confirmed':[], 'Deaths':[]}
    sorted_state_names = sorted(state_names)
    id_count = 0
    if submission_round == 2:
        for i in range(7):
            for state in sorted_state_names:
                if state in result:
                    submission['ForecastID'].append(id_count)
                    submission['Confirmed'].append(result[state]['confirmed'][i+14])
                    submission['Deaths'].append(result[state]['death'][i+14])
                    id_count += 1
    elif submission_round == 1:
        for i in range(predict_length):
            for state in sorted_state_names:
                if state in result:
                    submission['ForecastID'].append(id_count)
                    submission['Confirmed'].append(result[state]['confirmed'][i])
                    submission['Deaths'].append(result[state]['death'][i])
                    id_count += 1       

    # write to csv
    output_name = "arma_submission%d.csv" % submission_round
    with open(output_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(submission.keys())
        for i in range(id_count):
            cur_row = [submission['ForecastID'][i], submission['Confirmed'][i], submission['Deaths'][i]]
            writer.writerow(cur_row)
    print('done')


if __name__ == "__main__":
    main()
