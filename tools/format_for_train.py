import calendar
import pandas as pd
import numpy as np

def bar():
    print('*'*88)

def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    else:
        return .5 * (1 + np.tanh(.5 * x))
        
def count_to_frac(n):
    n = float(n)
    return 1.0/n
        

def main():
    df = pd.read_csv('data/train.csv',
        engine='c',
        low_memory=False,
        nrows=1000000
        )

    df.set_index('key')

    data = []

    for i, ds in df.iterrows():
        dsa = np.array(ds)
###############################################################################
###############################################################################
        date, time, _  = dsa[2].split(' ')
        year, month, day = date.split('-')
        year = int(year)
        month = int(month)
        day = int(day)
        hour, minute, _  = time.split(':')
        hour = int(hour)
        minute = int(minute)

        ndsa = []
        ndsa.append(calendar.weekday(year, month, day)/7.0) #weekday
        ndsa.append(hour/24.0) #hour of day
        ndsa.append(sigmoid(dsa[5])) #dropoff long
        ndsa.append(sigmoid(dsa[6])) #dropoff lat
        ndsa.append(sigmoid(dsa[5]-dsa[3])) #difference in longitude
        ndsa.append(sigmoid(dsa[6]-dsa[4])) #diff in latitude
        ndsa.append(dsa[1]) #price
###############################################################################
###############################################################################
        data.append(ndsa)

    df = pd.DataFrame(
            { "wday"                : [x[0] for x in data],
              "hour"                : [x[1] for x in data],
              "dropoff_longitude"   : [x[2] for x in data],
              "dropoff_latitude"    : [x[3] for x in data],
              "longitude"           : [x[4] for x in data],
              "latitude"            : [x[5] for x in data],
              "price"               : [x[6] for x in data]},
            index = list(range(len(data)))
            )

    df.to_csv('data/formated_train.csv')
###############################################################################
###############################################################################

if __name__ == '__main__':
    main()
