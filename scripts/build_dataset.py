from datetime import datetime
import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno


# conda install -c conda-forge missingno


#   time x y z magnitude n_steps avg_between_steps cadence min max mean median variance std_dev zero_crossing_rate
#   ok ok ok ok     ok       ok           ok         ok   ok  ok  ok    ok      ok      ok         ok
#
#   skewness kurtosis | TAC | is_intoxicated
#       ok       ok     ok

# make the index a Datetime object
def from_timestamp_to_datetime(timestamp):
    timestamp = float(timestamp) / 1000.0
    return datetime.fromtimestamp(float(timestamp))


# load the TAC as Series
tac = {}
pids = []
plots = []

with open('../data/pids.txt') as people:
    # extract pids to use as index for grouping by
    for pid in people:
        pids.append(pid[:-1])
        tac[pid[:-1]] = read_csv('../data/clean_tac/' + pid[:-1] + '_clean_TAC.csv',
                                 header=0,
                                 index_col=0,
                                 parse_dates=True,
                                 squeeze=True)
        tac[pid[:-1]].index = pd.to_datetime(tac[pid[:-1]].index, unit='s')

# load the csv
df = pd.read_csv('../data/all_accelerometer_data_pids_13.csv')
df_by_pid = {}

# group by pid and split into 13 dataframes
for pid_group in df.groupby(df['pid']):
    df_by_pid[pid_group[0]] = pid_group[1]
    df_by_pid[pid_group[0]].drop(['pid'], axis=1, inplace=True)

    # calculate magnitude of the (x,y,z) accelerations
    df_by_pid[pid_group[0]]['magnitude'] = np.sqrt(np.square(df_by_pid[pid_group[0]]['x']) +
                                                   np.square(df_by_pid[pid_group[0]]['y']) +
                                                   np.square(df_by_pid[pid_group[0]]['z']))
    df_by_pid[pid_group[0]].dropna(inplace=True)
    df_by_pid[pid_group[0]].drop_duplicates(inplace=True)

step_treshold = 1  # magnitude after which we can consider a step
# (steps) treshold=1 gives more than 1-2 steps per second but our magnitude is 3d so there could be movements
# other than stepping

interpolations = {}
df = pd.DataFrame()

for pid in pids:
    # make index datetimes
    df_by_pid[pid].index = df_by_pid[pid].time.apply(from_timestamp_to_datetime)

    # adding columns for each feature (some are calculated in more than one step
    #   e.g. calculate if magnitude is above step treshold ->
    #       -> calculate avg between a step and the one before ->
    #           -> calculate avg between avgs with rolling().agg(sum) for the total window
    df_by_pid[pid]['steps'] = np.where(df_by_pid[pid]['magnitude'] > step_treshold, 1, 0)
    df_by_pid[pid]['avg_between_steps'] = np.where(df_by_pid[pid]['steps'] > 0, df_by_pid[pid]['time'].diff(), 0)

    # zero crossings means how many time the acceleration for each ax changes sign
    df_by_pid[pid]['zero_crossings_x'] = np.where((np.signbit(df_by_pid[pid]['x'])), 1, 0)
    df_by_pid[pid]['zero_crossings_x'] = np.where((df_by_pid[pid]['zero_crossings_x'].diff(periods=-1) != 0), 1, 0)

    df_by_pid[pid]['zero_crossings_y'] = np.where((np.signbit(df_by_pid[pid]['y'])), 1, 0)
    df_by_pid[pid]['zero_crossings_y'] = np.where((df_by_pid[pid]['zero_crossings_y'].diff(periods=-1) != 0), 1, 0)

    df_by_pid[pid]['zero_crossings_z'] = np.where((np.signbit(df_by_pid[pid]['z'])), 1, 0)
    df_by_pid[pid]['zero_crossings_z'] = np.where((df_by_pid[pid]['zero_crossings_z'].diff(periods=-1) != 0), 1, 0)

    # cadence is speed expressed in steps per second
    df_by_pid[pid]['cadence'] = df_by_pid[pid][
        'steps']  # divide by time
    # with our 1 second window the cadence (steps per second) is just equal to the number of steps

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

    # resample every 0.2 seconds
    # we've done some tests to see how much each dataframe becomes smaller
    # 100L -> between 0.7 and 3.8 times smaller
    #   (0.7: dataset can became larger because resample adds rows for nan values as well)
    # 200L -> between 1.5 and 7 times smaller
    # 500L -> between 3.5 and 18 times smaller
    # 1000L -> between 7 and 38 times smaller
    df_by_pid[pid] = df_by_pid[pid].resample('200L').mean()

    # merge each dataframe with the respective tac series
    df_by_pid[pid] = pd.merge_asof(df_by_pid[pid], tac[pid].to_frame(),
                                   left_index=True, right_index=True,
                                   tolerance=pd.Timedelta("5ms"),
                                   # it looks like it doesn't affect merging but it works so i won't change it
                                   allow_exact_matches=True,
                                   direction='nearest')
    # performs interpolation between TAC readings
    # TODO maybe we can interpolate with different method and do some cross validation
    df_by_pid[pid]['TAC_Reading'].interpolate(method='linear', inplace=True, limit_direction='both')

    # plot and save plots to see TACs wrt magnitudes (mean of magnitudes in the window)
    # df_by_pid[pid].plot(y=[('magnitude', 'mean'), 'TAC_Reading'], use_index=True)
    # plt.savefig('../plots/interpolated/' + pid + '.pdf', bbox_inches='tight')

    # drop every nan value
    # info before drop
    # print('\n'+pid)
    # df_by_pid[pid].info()
    # msno.bar(df_by_pid[pid])
    # plt.savefig('../plots/bar/' + pid + '.pdf', bbox_inches='tight')
    # msno.matrix(df_by_pid[pid])
    # plt.savefig('../plots/matrix/' + pid + '.pdf', bbox_inches='tight')
    # msno.heatmap(df_by_pid[pid])
    # plt.savefig('../plots/heatmap/' + pid + '.pdf', bbox_inches='tight')
    # plt.show()

    df_by_pid[pid].dropna(inplace=True)
    # info after drop
    df_by_pid[pid].info()

    # restore the pid column
    df_by_pid[pid]['pid'] = pid

    # restore the original dataset stacking the grouped by pid ones
    df = pd.concat([df, df_by_pid[pid]], axis=0)
    # df.info()

# df.to_csv('../data/final_dataset_drop.csv')

# TODO JB3156 is weird, maybe we should remove it
# TODO how do we deal with the values dropped that leave a hole in the window that can be big?
