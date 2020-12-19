import csv
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from reference import data_process, calculate_from_diff

data_dir = os.getcwd() + "\\data\\state_diff\\"
lag = 5
predict_length = 26
output_name = 'ar_submission.csv'
state_names = ["Alabama", "Alaska", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District of Columbia", "Delaware", "Florida", "Georgia", 
"Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", 
"Montana", "North_Carolina", "North_Dakota", "Nebraska", "New_Hampshire", "New_Jersey", "New_Mexico", "Nevada", "New_York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto_Rico", 
"Rhode_Island", "South_Carolina", "South_Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin_Islands", "Vermont", "Washington", "Wisconsin", "West_Virginia", "Wyoming"]


def ar_fit_and_predict(train, lag, predict_length):
    """
    Find the parameters AR model by training data and predict future values
    """
    train_length = len(train)

    # fit parameters
    x_values = list()
    for i in range(lag+1, train_length):
        past_value = train[i-lag:i].tolist()
        past_value.append(1)
        x_values.append(past_value)
    y_values = train[lag+1:].tolist()
    lr = LinearRegression()
    lr.fit(x_values, y_values)

    # predict future values
    result = list()
    whole_values = train[:].tolist()
    for i in range(predict_length):
        x_value = whole_values[i+train_length-lag:i+train_length]
        x_value.append(1)
        predict_value = lr.predict([x_value])
        whole_values.append(predict_value[0])
        result.append(predict_value[0])

    return result


def make_ar_prediction(data,lag,predict_length):
    """
    Main function to create AR model and make prediction

    data: pandas dataframe that contains the training data
    lag: the number of lag days of AR model
    predict_length: the number of future values the model will predict
    """
    result = dict()

    diff_confirmed_predict = ar_fit_and_predict(data['confirmed_diff'], lag, predict_length)
    diff_death_predict = ar_fit_and_predict(data['death_diff'], lag, predict_length)

    result['confirmed'] = calculate_from_diff(data['Confirmed'][-1], diff_confirmed_predict)
    result['death'] = calculate_from_diff(data['Deaths'][-1], diff_death_predict)
    return result


def main():
    # run model
    result = dict()
    for state in state_names:
        data = data_process(state)
        if data != -1:
            result[state] = make_ar_prediction(data,lag,predict_length)

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
    with open(output_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(submission.keys())
        for i in range(id_count):
            cur_row = [submission['ForecastID'][i], submission['Confirmed'][i], submission['Deaths'][i]]
            writer.writerow(cur_row)
    print('done')


if __name__ == "__main__":
    main()