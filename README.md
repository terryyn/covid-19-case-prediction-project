# covid-19-case-prediction-project

Group Project for UCLA Data Mining Course

Use previous covid-19 positive and death cases to build time-series model to predict future case numbers for every state.
 
make prediction csv by the example submission format: Day1: Alabama, Alaska, ....    Day2: Alabama, ....

Environment Init:

    conda env create -f project_env.yml

    conda activate project_env


Data Preprocess:

    python -W ignore preprocess_diff.py filename(train.csv for example)


Prediction generated

    python -W ignore arma.py 1/2  (1 for submission1 2 for submission2)

    The result is arma_submission1.csv or arma_submission2.csv


This is just for submission1 to test the model, no need for submission2:

    Test step for checking the model: run "python test.py filename" to compute MAPE (modify the file directory inside before)


Note:

    between each submission run data preprocess first as it will write to same directory to overwrite data