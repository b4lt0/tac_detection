# ML project work by Jordi Segura Pons and Andrea Balillo

# The study decomposed each time series into 10 second windows and performed binary classification
# to predict if windows corresponded to an intoxicated participant (TAC >= 0.08) or sober participant (TAC < 0.08).
# The study tested several models and achieved a test accuracy of 77.5% with a random forest.

# Features: Three-axis time series accelerometer data
# Target: Time series transdermal alcohol content (TAC) data (real-time measure of intoxication).

from datetime import datetime
import matplotlib.dates as mdates
import matplotlib
from pandas import read_csv
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np

# Here I show how the TAC reading behaves
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

for pid in pids:
    for ts in series[pid].index:
        print(datetime.fromtimestamp(ts))

# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter)
dt_array = []
for pid, i in zip(pids, range(0, len(pids))):
    plots.append(plt.figure(i))
    ts_array = np.array(series[pid].index.values)
    for ts in ts_array:
        dt_array.append(datetime.fromtimestamp(ts).strftime('%H:%M'))
    plt.plot(dt_array, series[pid].values, marker='o')
    plt.ylim(0, 0.25)
    # TODO scale the x (time) axis, like make it equally distributed
    # TODO show the threshold 0.08
    plt.axhline(y=0.08, linewidth=1, color='k')
    ax = plt.gca()
    ax.set_xticks(ax.get_xticks()[::4])
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.show()
    dt_array = []

# TODO maybe we don't need all the tac readings but only the window near the 0.08 threshold
# TODO you have to equally distribute drunk and not drunk example
# TODO build pandas dataframe

# dataframe = DataFrame()
# dataframe['time'] = [series[pid].index[i]. for i in range(len(series))]

# print(dataframe.head(5))
