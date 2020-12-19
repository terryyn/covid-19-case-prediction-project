import pandas as pd
import numpy as np
import sys
import os

stateList = ['Alabama','Alaska','Arizona','Arkansas','California','Colorado',
             'Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho',
             'Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',
             'Maine','Maryland','Massachusetts','Michigan','Minnesota',
             'Mississippi','Missouri','Montana','Nebraska','Nevada',
             'New Hampshire','New Jersey','New Mexico','New York',
             'North Carolina','North Dakota','Ohio','Oklahoma','Oregon',
             'Pennsylvania','Rhode Island','South Carolina','South Dakota',
             'Tennessee','Texas','Utah','Vermont','Virginia','Washington',
             'West Virginia','Wisconsin','Wyoming']


def main():
    outdir = './state_diff'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    train_file = sys.argv[1]
    df = pd.read_csv (train_file)
    for state in stateList:
        df_mask=df['Province_State']==state
        cur_state=df[df_mask]
        cur_state['death_diff'] = (cur_state['Deaths'] - cur_state['Deaths'].shift(1))
        cur_state['confirmed_diff'] = (cur_state['Confirmed'] - cur_state['Confirmed'].shift(1))
        file_name = "state_diff/{}.csv".format(state).replace(' ', '_')
        cur_state.to_csv(file_name, index=False)


if __name__ == "__main__":
    main()