import calendar
import pandas as pd
import numpy as np

def bar():
    print('*'*88)

def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))
        
def count_to_frac(n):
    n = float(n)
    return 1.0/n

def main():
    df = pd.read_csv('data/test.csv')
    df.set_index('key')

    data = []

    for i, ds in df.iterrows():
        dsa = np.array(ds)
###############################################################################
###############################################################################
        date, time, _  = dsa[1].split(' ')
        year, month, day = date.split('-')
        year = int(year)
        month = int(month)
        day = int(day)
        hour, minute, _  = time.split(':')
        hour = int(hour)
        minute = int(minute)

        ndsa = []
        ndsa.append(month/12.0) #month  
        ndsa.append(day/31.0) #day of month
        ndsa.append(calendar.weekday(year, month, day)/7.0) #weekday
        ndsa.append(hour/24.0) #hour of day
        ndsa.append(minute/60.0) #minute of hour
        ndsa.append(sigmoid(dsa[2])) #pickup long
        ndsa.append(sigmoid(dsa[3])) #pickup lat
        ndsa.append(sigmoid(dsa[4])) #dropoff long
        ndsa.append(sigmoid(dsa[5])) #dropoff lat
        ndsa.append(sigmoid(dsa[4]-dsa[2])) #difference in longitude
        ndsa.append(sigmoid(dsa[5]-dsa[3])) #diff in latitude
        ndsa.append(count_to_frac(dsa[6]))
###############################################################################
###############################################################################
        data.append(ndsa)

    df = pd.DataFrame(
            { "month"               : [x[0] for x in data],
              "day"                 : [x[1] for x in data],
              "wday"                : [x[2] for x in data],
              "hour"                : [x[3] for x in data],
              "minute"              : [x[4] for x in data],
              "pickup_longitude"    : [x[5] for x in data],
              "pickup_latitude"     : [x[6] for x in data],
              "dropoff_longitude"   : [x[7] for x in data],
              "dropoff_latitude"    : [x[8] for x in data],
              "longitude"           : [x[9] for x in data],
              "latitude"            : [x[10] for x in data],
              "pass_count"          : [x[11] for x in data]},
            index = list(range(len(data)))
            )

    df.to_csv('data/formated_test.csv')
###############################################################################
###############################################################################

if __name__ == '__main__':
    main()
