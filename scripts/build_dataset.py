from datetime import datetime
import pandas as pd
from pandas import read_csv
import numpy as np


#   time x y z magnitude n_steps avg_between_steps cadence min max mean median variance std_dev zero_crossing_rate
#   ok ok ok ok     ok       ok           ok         ok   ok  ok  ok    ok      ok      ok         ok
#
#   skewness kurtosis | TAC | is_intoxicated
#       ok       ok     #TODO add TAC from interpolation


def from_timestamp_to_datetime(timestamp):
    timestamp = float(timestamp) / 1000.0
    return datetime.fromtimestamp(float(timestamp))


series = {}
pids = []
plots = []
with open('../data/pids.txt') as people:
    for pid in people:
        pids.append(pid[:-1])
        series[pid[:-1]] = read_csv('../data/clean_tac/' + pid[:-1] + '_clean_TAC.csv',
                                    header=0,
                                    index_col=0,
                                    parse_dates=True,
                                    squeeze=True)
df = pd.read_csv('../data/all_accelerometer_data_pids_13.csv')
df_by_pid = {}
for pid_group in df.groupby(df['pid']):
    df_by_pid[pid_group[0]] = pid_group[1]
    df_by_pid[pid_group[0]].drop(['pid'], axis=1, inplace=True)
    df_by_pid[pid_group[0]]['magnitude'] = np.sqrt(np.square(df_by_pid[pid_group[0]]['x']) +
                                                   np.square(df_by_pid[pid_group[0]]['y']) +
                                                   np.square(df_by_pid[pid_group[0]]['z']))
    df_by_pid[pid_group[0]].dropna(inplace=True)
    df_by_pid[pid_group[0]].drop_duplicates(inplace=True)

step_treshold = 1  # magnitude after which we can consider a step
# (steps) treshold=1 gives more than 1-2 steps per second but our magnitude is 3d so there could be movements
# other than stepping

for pid in pids:
    df_by_pid[pid] = df_by_pid[pid]
    df_by_pid[pid].index = df_by_pid[pid].time.apply(from_timestamp_to_datetime)

    # print(pid)
    # print(df_by_pid[pid].head())

    df_by_pid[pid]['steps'] = np.where(df_by_pid[pid]['magnitude'] > step_treshold, 1, 0)
    df_by_pid[pid]['avg_between_steps'] = np.where(df_by_pid[pid]['steps'] > 0, df_by_pid[pid]['time'].diff(), 0)

    df_by_pid[pid]['zero_crossings_x'] = np.where((np.signbit(df_by_pid[pid]['x'])), 1, 0)
    df_by_pid[pid]['zero_crossings_x'] = np.where((df_by_pid[pid]['zero_crossings_x'].diff(periods=-1) != 0), 1, 0)

    df_by_pid[pid]['zero_crossings_y'] = np.where((np.signbit(df_by_pid[pid]['y'])), 1, 0)
    df_by_pid[pid]['zero_crossings_y'] = np.where((df_by_pid[pid]['zero_crossings_y'].diff(periods=-1) != 0), 1, 0)

    df_by_pid[pid]['zero_crossings_z'] = np.where((np.signbit(df_by_pid[pid]['z'])), 1, 0)
    df_by_pid[pid]['zero_crossings_z'] = np.where((df_by_pid[pid]['zero_crossings_z'].diff(periods=-1) != 0), 1, 0)

    df_by_pid[pid]['cadence'] = df_by_pid[pid][
        'steps']  # with our 1 second window the cadence (steps per second) is just equal to the number of steps

    df_by_pid[pid] = df_by_pid[pid].rolling('1000L').agg(
        {'x': ['min', 'max', 'mean', 'median', 'var', 'std', 'skew', 'kurt'],
         'y': ['min', 'max', 'mean', 'median', 'var', 'std', 'skew', 'kurt'],
         'z': ['min', 'max', 'mean', 'median', 'var', 'std', 'skew', 'kurt'],
         'magnitude': ['min', 'max', 'mean', 'median', 'var', 'std', 'skew', 'kurt'],
         'steps': ['sum'],
         'avg_between_steps': ['mean'],
         'zero_crossings_x': ['sum'],
         'cadence': ['sum']}
    )

    df_by_pid[pid].dropna(inplace=True)
    df_by_pid[pid].drop_duplicates(inplace=True)

    # 100L -> between 0.7 and 3.8 times smaller
    #   (0.7: dataset can became larger because resample adds rows for nan values as well)
    # 200L -> between 1.5 and 7 times smaller
    # 500L -> between 3.5 and 18 times smaller
    # 1000L -> between 7 and 38 times smaller
    df_by_pid[pid] = df_by_pid[pid].resample('200L').mean()


    # print(df_by_pid[pid].head())
    # print()
