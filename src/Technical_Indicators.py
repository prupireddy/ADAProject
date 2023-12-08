import pandas as pd    """
    This will impute the y-value of an x coordinate based on the linear interpolation built from the two points given to it.

    :first_x: x-coordinate of the first point used to build the interpolator
    :first_y: y-coordinate of the first point used to build the interpolator
    :second_x: x-coordinate of the second point used to build the interpolator
    :second_y: y-coordinate of the second point used to build the interpolator
    :arg: x-coordinate of the point for which the y-value needs to be interpolated
    :return: interpolated y-coordinate of the point x-coordinate of which was arg
    """
import matplotlib.pyplot as plt
import numpy as np
import sys

def linear_interpolator(first_x, first_y, second_x, second_y, arg):

    return (first_y + (arg - first_x)*((second_y - first_y)/(second_x - first_x)))


def calculate_rsi(data, column_name='Close', n=14):
    """
    This will calculate the rsi value (very common technical indicator) for all time steps of the data given to it.

    :data: dataset that contains the data that is to be analyzed
    :column_name: refers to the column of the dataset that is to be analyzed
    :n: this is a parameter for the rsi calculation
    :returns: returns the rsi value for all rows 
    """
    delta = data[column_name].diff(1)

    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    avg_gain = gains.rolling(window=n, min_periods=1).mean()
    avg_loss = losses.rolling(window=n, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rs[avg_loss == 0] = np.inf

    rsi = 1 - (1 / (1 + rs))

    return rsi

def rsi_wrapper(rsi_val):
    """
    This will calculate IMX value and branch given an rsi value

    :rsi_val: this is the rsi value that the the wrapper is analyzing
    :returns: returns the imx, branch for that rsi value 
    """
    if rsi_val < .3 and rsi_val >= 0:
        return linear_interpolator(.3, .7, 0, 1, rsi_val), 0
    elif rsi_val <= .7 and rsi_val >= .3:
        return 0, 1
    elif rsi_val > .7 and rsi_val <= 1:
        return linear_interpolator(.7, -.7, 1, -1, rsi_val), 2
    else: 
        print(rsi_val)
        print("Out of Range")
        sys.exit(1)

def calculate_roc(data, column_name='Close', n=1):
    """
    This will calculate the roc value (rate of change - ratio of price at current time step to a part time step) for all time steps of the data given to it.

    :data: dataset that contains the data that is to be analyzed
    :column_name: refers to the column of the dataset that is to be analyzed
    :n: this is a parameter defining the number of time steps beforehand from which the denominator will be formed
    :returns: returns the roc value for all rows 
    """
    roc = ((data[column_name]) / data[column_name].shift(n)) 

    return roc

def roc_wrapper(roc_val):
    """
    This will calculate IMX value and branch given an roc value

    :roc_val: this is the roc value that the the wrapper is analyzing
    :returns: returns the imx, branch for that roc value 
    """
    if roc_val < .9:
        return -1, 0
    elif roc_val <= 1.1 and roc_val >= .9:
        return 0, 1
    elif roc_val > 1.1:
        return 1, 2
    else: 
        print("Out of Range")
        sys.exit(1)


def rci_wrapper(rci_val):
    """
    This will calculate IMX value and branch given an rci value.

    :rci_val: this is the rci value that the the wrapper is analyzing
    :returns: returns the imx, branch for that rci value 
    """
    if rci_val < -.7 and rci_val >= -1:
        return linear_interpolator(-.7, .7, -1, 1, rci_val), 0
    elif rci_val <= .7 and rci_val >= -.7:
        return 0, 1
    elif rci_val > .7 and rci_val <= 1:
        return linear_interpolator(.7, -.7, 1, -1, rci_val), 2
    else: 
        print("Out of Range")
        sys.exit(1)
    
def calculate_volume_ratio(data, price_column='Close', volume_column='Volume', period=20):
    """
    This will calculate the volume ratio  (the sum of the trading volume on days when stock prices rose
    during a certain period and half the trading volume on days when stock prices
    remained unchanged was divided by the total trading volume for that period) for all time steps of the data given to it.

    :data: dataset that contains the data that is to be analyzed
    :price_column: refers to the column of the dataset that has the price data
    :volume_name: refers to the column of the dataset that contains the volume information
    :n: this is a parameter for the roc calculation
    :returns: returns the roc value for all rows 
    """
    data['Price Change'] = data[price_column].diff()

    data['Up Days'] = data['Price Change'].apply(lambda x: 1 if x > 0 else 0)
    data['Unchanged Days'] = data['Price Change'].apply(lambda x: 1 if x == 0 else 0)

    data['Numerator'] = data['Up Days'] * data[volume_column] + 0.5 * data['Unchanged Days'] * data[volume_column]
    data['Numerator'] = data['Numerator'].rolling(window=period, min_periods=1).sum()

    data['Denominator'] = data[volume_column].rolling(window=period, min_periods=1).sum()

    data[f'volume_{period}'] = data['Numerator'] / data['Denominator']

    data = data.drop(['Up Days', 'Unchanged Days', 'Numerator', 'Denominator'], axis=1)

    return data

def volume_wrapper(volume_val):
    """
    This will calculate IMX value and branch given an volume ratio value

    :volume_val: this is the volume value that the the wrapper is analyzing
    :returns: returns the imx, branch for that volume value 
    """
    if volume_val < .25 and volume_val >= 0:
        return linear_interpolator(.25, .8, 0, 1, volume_val), 0
    elif volume_val <= .75 and volume_val >= .25:
        return 0, 1
    elif volume_val > .75 and volume_val <= 1:
        return linear_interpolator(.75, -.8, 1, -1, volume_val), 2
    else: 
        print("Out of Range")
        sys.exit(1)

def calculate_rate_of_deviation(data, price_column='Close', moving_average_period=20):
    """
    This will calculate the rate of deviation (percentage error from moving average) for all time steps of the data given to it.

    :data: dataset that contains the data that is to be analyzed
    :price_column: refers to the column of the dataset that has the price data
    :moving_average_period: this controls the value of the window for the moving average
    :returns: returns the rod value for all rows 
    """
    data['Moving Average'] = data[price_column].rolling(window=moving_average_period, min_periods=1).mean()

    data[f'rod_{moving_average_period}'] = (data[price_column] - data['Moving Average']) / data['Moving Average']

    data = data.drop(['Moving Average'], axis=1)

    return data

def rod_wrapper(rod_val):
    """
    This will calculate IMX value and branch given a rate of deviation value

    :volume_val: this is the rate of deviation value that the the wrapper is analyzing
    :returns: returns the imx, branch for that rod value 
    """
    if rod_val < -.1:
        return 1, 0
    elif rod_val <= -0.05 and rod_val >= -.1:
        return linear_interpolator(-.1, 1, -.05, .5, rod_val), 1
    elif rod_val >  -0.05 and rod_val <=  0.05:
        return 0, 2
    elif rod_val >  0.05 and rod_val <=  .1:
        return linear_interpolator(.1, -1, .05, -.5, rod_val), 3
    elif rod_val > .1:
        return -1, 4
    else: 
        print("Out of Range")
        sys.exit(1)
   
def calculate_stochastic_kd(data, close_column='Close', low_column='Low', high_column='High', lookback_days=14, smoothing_period=3):
    """
    This will first calculate the k% stochastic, which is the ratio of the lowest price from the previous lookback_days subtracted from the current price divided by the lowest price 
    subtracted from the high for all the previous lookback days for all time steps of the data given to it. Then, it calculates the moving average of the %k statistic as the %d 
    statistic.

    :data: dataset that contains the data that is to be analyzed
    :close_column: refers to the column of the dataset that has the column data
    :moving_average_period: this controls the value of the window for the moving average used to calculate the %k statistic
    :smoothing period: this control the length of the period used for the window over the %k statistic which is how %d is calculated 
    :returns: returns the d% stochastic value for all rows 
    """
    
    data['Lowest Low'] = data[low_column].rolling(window=lookback_days, min_periods=1).min()
    data['Highest High'] = data[high_column].rolling(window=lookback_days, min_periods=1).max()

    data['%K'] = ((data[close_column] - data['Lowest Low']) / (data['Highest High'] - data['Lowest Low'])) * 100

    data[f'd_{lookback_days}'] = data['%K'].rolling(window=smoothing_period, min_periods=1).mean()

    data = data.drop(['Lowest Low', 'Highest High','%K'], axis=1)
    
    return data
    
def kd_wrapper(kd_val):
    """
    This will calculate IMX value and branch given a kd value

    :volume_val: this is kd value that the the wrapper is analyzing
    :returns: returns the imx, branch for that rod value 
    """
    if kd_val < 30 and kd_val >= 0:
        return linear_interpolator(30, .7, 0, 1, kd_val), 0
    elif kd_val <= 70 and kd_val >= 30:
        return 0, 1
    elif kd_val >  70 and kd_val <= 100:
        return linear_interpolator(70, -.7, 100, -1, kd_val), 2
    else: 
        print("Out of Range")
        sys.exit(1)

def calculate_golden_dead_cross(data, short_term=5, long_term=26, price_column='Close'):
    """
    This will calculate whether there is a golden or dead cross (whether or not the shorter moving average crosses the longer moving average from
    below or above respectively) on all time steps of the data given to it.

    :data: dataset that contains the data that is to be analyzed
    :short_term: windows for the smaller moving average
    :short_term: windows for the longer-term moving average
    :price_column: refers to the column of the dataset that has the price data
    :returns: returns a 1 for the IMX value (the next program will add one to this for the branch number) for three days when a golden cross occurs and -1 for a dead cross
    """
    
    data[f'Short_MA_{short_term}'] = data[price_column].rolling(window=short_term, min_periods=1).mean()
    data[f'Long_MA_{long_term}'] = data[price_column].rolling(window=long_term, min_periods=1).mean()

    data['gd_cross'] = 0  # 0 represents no signal

    golden_cross_mask = (data[f'Short_MA_{short_term}'] > data[f'Long_MA_{long_term}']) & \
                        (data[f'Short_MA_{short_term}'].shift(1) <= data[f'Long_MA_{long_term}'].shift(1))
    data.loc[golden_cross_mask, 'gd_cross'] = 1
    print("hit1")

    dead_cross_mask = (data[f'Short_MA_{short_term}'] < data[f'Long_MA_{long_term}']) & \
                      (data[f'Short_MA_{short_term}'].shift(1) >= data[f'Long_MA_{long_term}'].shift(1))
    data.loc[dead_cross_mask, 'gd_cross'] = -1
    print("hit1")

    data = data.drop([f'Short_MA_{short_term}', f'Long_MA_{long_term}'], axis=1)

    return data

def calculate_macd(data, short_term=5, long_term=26, signal_period=9, price_column='Close'):
    """
    This will calculate the macd value (whether the moving average of the difference of two moving averages crosses the difference of the two moving averages) of all time steps of the data given to it. 

    :data: dataset that contains the data that is to be analyzed
    :short_term: windows for the smaller moving average used in difference
    :long_term: windows for the longer-term moving average used in difference
    :signal_period: window length to calculate the moving average of the difference
    :price_column: refers to the column of the dataset that has the price data
    :returns: returns a 1 for the IMX value (the next program will add one to this for the branch number) for three days for a buy outcome and -1 for a sell outcome.
    """

    data[f'Short_MA_{short_term}'] = data[price_column].rolling(window=short_term, min_periods=1).mean()
    data[f'Long_MA_{long_term}'] = data[price_column].rolling(window=long_term, min_periods=1).mean()

    data['MACD'] = data[f'Short_MA_{short_term}'] - data[f'Long_MA_{long_term}']

    data['Signal'] = data['MACD'].rolling(window=signal_period, min_periods=1).mean()

    data['macd_signal'] = 0

    buy_signal_mask = (data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1))
    data.loc[buy_signal_mask, 'macd_signal'] = 1
    print("hit2")

    sell_signal_mask = (data['MACD'] < data['Signal']) & (data['MACD'].shift(1) >= data['Signal'].shift(1))
    data.loc[sell_signal_mask, 'macd_signal'] = -1
    print("hit2")

    data = data.drop([f'Short_MA_{short_term}', f'Long_MA_{long_term}', 'Signal'], axis=1)

    return data

#the rest of the code actually performs the application of the technical indicator value and its respective IMX/branch values
df = pd.read_csv('SPY.csv')

df['rsi_5'] = calculate_rsi(df, n = 5)
df['rsi_13'] = calculate_rsi(df, n = 13)
df['rsi_26'] = calculate_rsi(df, n = 26)
df[['rsi_5_IMX', 'rsi_5_branch']] = df['rsi_5'].apply(lambda x: pd.Series(rsi_wrapper(x)))
df[['rsi_13_IMX', 'rsi_13_branch']] = df['rsi_13'].apply(lambda x: pd.Series(rsi_wrapper(x)))
df[['rsi_26_IMX', 'rsi_26_branch']] = df['rsi_26'].apply(lambda x: pd.Series(rsi_wrapper(x)))

df['roc_5'] = calculate_roc(df, n = 5)
df['roc_13'] = calculate_roc(df, n = 13)
df['roc_26'] = calculate_roc(df, n = 26)
df.drop(index=range(26), inplace = True)
df = df.reset_index(drop=True)
df[['roc_5_IMX', 'roc_5_branch']] = df['roc_5'].apply(lambda x: pd.Series(roc_wrapper(x)))
df[['roc_13_IMX', 'roc_13_branch']] = df['roc_13'].apply(lambda x: pd.Series(roc_wrapper(x)))
df[['roc_26_IMX', 'roc_26_branch']] = df['roc_26'].apply(lambda x: pd.Series(roc_wrapper(x)))

df['rci_9'] = df['Close'].rolling(window=9).apply(lambda x: pd.Series(x).rank().corr(pd.Series(range(1, 9 + 1))), raw=True)
df['rci_18'] = df['Close'].rolling(window=18).apply(lambda x: pd.Series(x).rank().corr(pd.Series(range(1, 18 + 1))), raw=True)
df['rci_27'] = df['Close'].rolling(window=27).apply(lambda x: pd.Series(x).rank().corr(pd.Series(range(1, 27 + 1))), raw=True)
df.drop(index=range(27), inplace = True)
df[['rci_9_IMX', 'rci_9_branch']] = df['rci_9'].apply(lambda x: pd.Series(rci_wrapper(x)))
df[['rci_18_IMX', 'rci_18_branch']] = df['rci_18'].apply(lambda x: pd.Series(rci_wrapper(x)))
df[['rci_27_IMX', 'rci_27_branch']] = df['rci_27'].apply(lambda x: pd.Series(rci_wrapper(x)))

df = calculate_volume_ratio(df, period = 5)
df = calculate_volume_ratio(df, period = 13)
df = calculate_volume_ratio(df, period = 26)
df[['volume_5_IMX', 'volume_5_branch']] = df['volume_5'].apply(lambda x: pd.Series(volume_wrapper(x)))
df[['volume_13_IMX', 'volume_13_branch']] = df['volume_13'].apply(lambda x: pd.Series(volume_wrapper(x)))
df[['volume_26_IMX', 'volume_26_branch']] = df['volume_26'].apply(lambda x: pd.Series(volume_wrapper(x)))

df = calculate_rate_of_deviation(df, moving_average_period = 5)
df = calculate_rate_of_deviation(df, moving_average_period = 13)
df = calculate_rate_of_deviation(df, moving_average_period = 26)
df[['rod_5_IMX', 'rod_5_branch']] = df['rod_5'].apply(lambda x: pd.Series(rod_wrapper(x)))
df[['rod_13_IMX', 'rod_13_branch']] = df['rod_13'].apply(lambda x: pd.Series(rod_wrapper(x)))
df[['rod_26_IMX', 'rod_26_branch']] = df['rod_26'].apply(lambda x: pd.Series(rod_wrapper(x)))

df = calculate_stochastic_kd(df, lookback_days = 12)
df = calculate_stochastic_kd(df, lookback_days = 20)
df = calculate_stochastic_kd(df, lookback_days = 30)
df[['d_12_IMX', 'd_12_branch']] = df['d_12'].apply(lambda x: pd.Series(kd_wrapper(x)))
df[['d_20_IMX', 'd_20_branch']] = df['d_20'].apply(lambda x: pd.Series(kd_wrapper(x)))
df[['d_30_IMX', 'd_30_branch']] = df['d_30'].apply(lambda x: pd.Series(kd_wrapper(x)))

df = calculate_golden_dead_cross(df)
df = calculate_macd(df)


df.to_csv('SPY_processed.csv')


print(df.columns)
