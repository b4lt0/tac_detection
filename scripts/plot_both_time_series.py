import matplotlib
matplotlib.use('Qt5Agg')

from datetime import datetime
from pandas import read_csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Here I show how the accelerometer reading behaves
from sklearn import preprocessing

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
# anche la fase del vettore(????)

dt_array_s = []
dt_array_ms = []

for pid, i in zip(pids, range(0, len(pids))):
    dt_array_s = []
    dt_array_ms = []

    plots.append(plt.figure(i))
    print(pid)
    ts_array_ms = np.array(df_by_pid[pid].time.array)
    ts_array_s = np.array(series[pid].index.values)
    # here i want to understand why TAC and magnitude are shifted, the delay between the first TAC reading and x,y,z reading is circa 5 hours
    print('TAC ' + str(ts_array_s[0]))
    print('MAG ' + str(ts_array_ms[0]))
    print('MAG /1000 ' + str(int(ts_array_ms[0] / 1000)))

    print('TAC ' + datetime.fromtimestamp(ts_array_s[0]).strftime('%H:%M'))
    # print('MAG '+datetime.fromtimestamp(ts_array_ms[0]).strftime('%H:%M'))
    print('MAG /1000 ' + datetime.fromtimestamp(int(ts_array_ms[0] / 1000)).strftime('%H:%M'))

    print('TAC-MAG ' + str(abs(ts_array_s[0] - ts_array_ms[0])))
    # print('TAC-MAG '+datetime.fromtimestamp(int(abs(ts_array_s[0]-ts_array_ms[0]))).strftime('%H:%M'))

    ts_array_ms = np.divide(ts_array_ms, 1000)
    print('TAC-MAG /1000 ' + str(abs(ts_array_s[0] - ts_array_ms[0])))
    print('TAC-MAG /1000 ' + datetime.fromtimestamp(int(abs(ts_array_s[0] - ts_array_ms[0]))).strftime('%H:%M'))
    print()
    '''
    for ts in ts_array_ms:
        dt_array_ms.append(datetime.fromtimestamp(int(ts / 1000)).strftime('%H:%M'))
    for ts in ts_array_s:
        dt_array_s.append(datetime.fromtimestamp(ts).strftime('%H:%M'))'''

    values_magnitude = df_by_pid[pid]['magnitude'].values
    values_magnitude = values_magnitude.reshape((len(values_magnitude), 1))
    scaler_magnitude = preprocessing.MinMaxScaler()
    # scaler_magnitude = scaler_magnitude.fit(values_magnitude)
    # print(pid+' Mean: %f, Variance: %f' % (scaler_magnitude.mean_, scaler_magnitude.var_))
    normalized_magnitude = scaler_magnitude.fit_transform(values_magnitude)
    '''for a in range(5):
        print(normalized_magnitude[a])'''

    # plt.plot(dt_array_ms, normalized_magnitude, color='b')
    plt.plot(ts_array_ms, normalized_magnitude, color='b')

    values_TAC = series[pid].values
    values_TAC = values_TAC.reshape((len(values_TAC), 1))
    scaler_TAC = preprocessing.MinMaxScaler()
    # scaler_TAC = scaler_TAC.fit(values_TAC)
    # print(pid+' Mean: %f, Variance: %f' % (scaler_TAC.mean_, scaler_TAC.var_))
    normalized_TAC = scaler_TAC.fit_transform(values_TAC)
    '''for b in range(5):
        print(normalized_TAC[b]+' '+values_TAC[b])'''

    # plt.plot(dt_array_s, normalized_TAC, color='r', marker='o')
    plt.plot(ts_array_s, normalized_TAC, color='r', marker='o')
    normalized_threshold = scaler_TAC.transform(np.array([0.08]).reshape(1, -1))
    plt.axhline(y=normalized_threshold, linewidth=1, color='k')

    ax = plt.gca()
    ax.set_xticks(ax.get_xticks()[::50])
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.title(pid)
    plt.legend(['magnitude', 'TAC readings', 'TAC limit'], loc='upper right')
    plt.savefig('../plots/TAC_magnitude/' + pid + '.pdf', bbox_inches='tight')
    plt.show()

'''
this is for creating the autocorrelation plots
from statsmodels.graphics.tsaplots import plot_acf
ac_ts = {}
plots = {}
for pid in pids:
    ac_ts[pid] = pd.Series(df_by_pid[pid]['magnitude'].values, index=df_by_pid[pid]['time'])
    plots[pid]=(plot_acf(ac_ts[pid], lags=20000, title=str(pid), fft=True, use_vlines=False))
    plots[pid].savefig('../plots/autocorrelations/' + pid + '.pdf', bbox_inches='tight')
    
    '''