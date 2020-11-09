import csv
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


# by tuning seems that using diff value with lag 29 is the best AR model
data_dir = "C:\\Users\\yeyun\\Google Drive\\university\\2020 fall\\CS 145\\covid-19-case-prediction-project\\data\\state_diff\\"
k = 10
lag = 29
predict_length = 26
state_names = ["Alabama", "Alaska", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", 
"Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", 
"Montana", "North_Carolina", "North_Dakota", "Nebraska", "New_Hampshire", "New_Jersey", "New_Mexico", "Nevada", "New_York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto_Rico", 
"Rhode_Island", "South_Carolina", "South_Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin_Islands", "Vermont", "Washington", "Wisconsin", "West_Virginia", "Wyoming"]


def mean_absolute_percentage_error(y_true, y_pred): 
    """
    Calculate MAPE as sklearn.metrics seems not updated to have it
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def calcualte_from_diff(last, diffs):
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


def make_regression_model_validation(data, k, lag, state):
    """
    make regression model from the data and use last k days of data as quick validation for picking right model

    Args:
        data: the numpy array
        k: the number of days as validation
        lag: the number of lag days used for AR
        state: the state the model for
    """
    result = dict()

    # extract diff value and split
    death_diff_train = data['death_diff'][:-k]
    confirmed_diff_train = data['confirmed_diff'][:-k]
    confirmed_train = data['Confirmed'][:-k]
    death_train = data['Deaths'][:-k] 
    confirmed_val = data['Confirmed'][-k:]
    death_val = data['Deaths'][-k:]

    # put data into AutoRegression model
    original_confirmed_model = AutoReg(confirmed_train, lags=lag).fit()
    original_death_model = AutoReg(death_train, lags=lag).fit()
    diff_confirmed_model = AutoReg(confirmed_diff_train, lags=lag).fit()
    diff_death_model = AutoReg(death_diff_train, lags=lag).fit()

    # predicts validation data
    confirmed_predict_val = original_confirmed_model.predict(start=len(confirmed_train), end=len(confirmed_train)+k-1,dynamic=False)
    death_predict_val = original_death_model.predict(start=len(death_train), end=len(death_train)+k-1,dynamic=False)
    diff_confirmed_predict_val = diff_confirmed_model.predict(start=len(confirmed_train), end=len(confirmed_train)+k-1,dynamic=False)
    diff_death_predict_val = diff_death_model.predict(start=len(death_train), end=len(death_train)+k-1,dynamic=False)
    confirmed_predict_val_diff = calcualte_from_diff(confirmed_train[-1], diff_confirmed_predict_val)
    death_predict_val_diff = calcualte_from_diff(death_train[-1], diff_death_predict_val)

    # calculate MAPE for validation 
    death_mape = mean_absolute_percentage_error(death_val,death_predict_val)
    confirmed_mape = mean_absolute_percentage_error(confirmed_val,confirmed_predict_val)
    diff_death_mape = mean_absolute_percentage_error(death_val,death_predict_val_diff)
    diff_confirmed_mape = mean_absolute_percentage_error(confirmed_val,confirmed_predict_val_diff)

    return 0


def make_regression_model_and_predict(data, lag, predict_length):
    """
    make regression model for test without validation split
    
    Args:
        data: the data to operate on
        lag: the number of lag days for AR
        predict_length: the number of days to predict

    Returns:
        result: dictionary of predicted result for confirmed and death
    """

    #init model
    result = dict()
    confirmed_model = AutoReg(data['confirmed_diff'], lags=lag).fit()
    death_model = AutoReg(data['death_diff'], lags=lag).fit()
    training_length = len(data['confirmed_diff'])

    # predict future diff values and calculate corresponding values
    diff_confirmed_predict = confirmed_model.predict(start=training_length, end=training_length+predict_length-1, dynamic=False)
    diff_death_predict = death_model.predict(start=training_length, end=training_length+predict_length-1, dynamic=False)
    confirmed_predict = calcualte_from_diff(data['Confirmed'][-1], diff_confirmed_predict)
    death_predict = calcualte_from_diff(data['Deaths'][-1], diff_death_predict)

    # store into result
    result['confirmed'] = confirmed_predict
    result['death'] = death_predict
    return result


def main():

    # run model
    result = dict()
    for state in state_names:
        data = data_process(state)
        if data != -1:
            result[state] = make_regression_model_and_predict(data,lag,predict_length)

    # generate submission dictionary
    submission = {'ForecastID':[], 'Confirmed':[], 'Deaths':[]}
    sorted_state_names = sorted(state_names)
    id_count = 0
    for i in range(predict_length):
        for state in sorted_state_names:
            if state in result:
                submission['ForecastID'].append(id_count)
                submission['Confirmed'].append(result[state]['confirmed'][i])
                submission['Deaths'].append(result[state]['death'][i])
                id_count += 1

    # write to csv
    with open('ar_submission.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(submission.keys())
        for i in range(id_count):
            cur_row = [submission['ForecastID'][i], submission['Confirmed'][i], submission['Deaths'][i]]
            writer.writerow(cur_row)
    print('done')



if __name__ == "__main__":
    main()