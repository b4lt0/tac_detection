from datetime import datetime
import matplotlib.dates as mdates
import matplotlib
from pandas import read_csv
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np

# Here I show how the accelerometer reading behaves
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

# TODO aggiungere una colonna di dati sul consumo di calorie ???? (serirebbe peso e altezza quindi  non va bene)
# TODO aggiungere una feature con la magnitude del vettore sqrt(x^2+y^2+z^2) FATTO
# a questo punto inserirei anche la fase del vettore(???????)

df_by_pid['CC6740'].drop(1, inplace=True)
df_by_pid['JB3156'].drop(0, inplace=True)
'''for pid in pids:
...     for ts in df_by_pid[pid].time.array:
...          print(datetime.fromtimestamp(int(ts/1000)))
...          i+=1
...          if i>10:
...              i=0
...              break'''

dt_array = []
df_stat = {}
for pid, i in zip(pids, range(0, len(pids))):
    # stats
    df_stat[pid] = pd.DataFrame(np.array(df_by_pid[pid].time.array)).describe()
    print(pid + ': \n' + datetime.fromtimestamp(int(df_stat[pid].__array__()[3] / 1000)).strftime('%m/%d/%Y, %H:%M:%S'))
    print(datetime.fromtimestamp(int(df_stat[pid].__array__()[7] / 1000)).strftime('%m/%d/%Y, %H:%M:%S'))

    # plots
    plots.append(plt.figure(i))
    ts_array = np.array(df_by_pid[pid].time.array)
    for ts in ts_array:
        dt_array.append(datetime.fromtimestamp(int(ts / 1000)).strftime('%H:%M'))
    plt.plot(dt_array, df_by_pid[pid]['magnitude'])
    # plt.ylim(0, 0.25)
    ax = plt.gca()
    ax.set_xticks(ax.get_xticks()[::50])
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.show()
    dt_array = []

# print(dataframe.head(5))
