import pandas as pd
import sys

def linear_interpolator(first_x, first_y, second_x, second_y, arg):
    return (first_y + (arg - first_x)*((second_y - first_y)/(second_x - first_x)))


def calculate_rsi(data, column_name='Close', n=14):

    delta = data[column_name].diff(1)

    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    avg_gain = gains.rolling(window=n, min_periods=1).mean()
    avg_loss = losses.rolling(window=n, min_periods=1).mean()

    rs = avg_gain / avg_loss

    rsi = 1 - (1 / (1 + rs))

    return rsi

def rsi_IMX_wrapper(rsi_val):
    if rsi_val < .3 and rsi_val >= 0:
        return linear_interpolator(.3, .7, 0, 1, rsi_val), 0
    elif rsi_val <= .7 and rsi_val >= .3:
        return 0, 1
    elif rsi_val > .7 and rsi_val <= 1:
        return linear_interpolator(.7, -.7, 1, -1, rsi_val), 2
    else: 
        print("Out of Range")
        sys.exit(1)

def calculate_roc(data, column_name='Close', n=1):

    roc = ((data[column_name]) / data[column_name].shift(n)) 

    return roc

def roc_IMX_wrapper(roc_val):
    if roc_val < .9:
        return -1, 0
    elif roc_val <= 1.1 and rsi_val >= .9:
        return 0, 1
    elif rsi_val > 1.1:
        return 1, 2
    else: 
        print("Out of Range")
        sys.exit(1)

df['RCI'] = df['Close'].rolling(window=window_size, center=True).apply(lambda x: pd.Series(x).rank().corr(pd.Series(range(1, window_size + 1))), raw=True)

def rci_IMX_wrapper(rci_val):
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

    data['Price Change'] = data[price_column].diff()

    data['Up Days'] = data['Price Change'].apply(lambda x: 1 if x > 0 else 0)
    data['Unchanged Days'] = data['Price Change'].apply(lambda x: 1 if x == 0 else 0)

    data['Numerator'] = data['Up Days'] * data[volume_column] + 0.5 * data['Unchanged Days'] * data[volume_column]
    data['Numerator'] = data['Numerator'].rolling(window=period, min_periods=1).sum()

    data['Denominator'] = data[volume_column].rolling(window=period, min_periods=1).sum()

    data['Volume Ratio'] = data['Numerator'] / data['Denominator']

    data = data.drop(['Up Days', 'Down Days', 'Unchanged Days', 'Numerator', 'Denominator'], axis=1)

    return data

def volume_IMX_wrapper(volume_val):
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

    data['Moving Average'] = data[price_column].rolling(window=moving_average_period, min_periods=1).mean()

    data['Rate of Deviation'] = (data[price_column] - data['Moving Average']) / data['Moving Average']

    data = data.drop(['Moving Average'], axis=1)

    return data

 def rod_IMX_wrapper(rod_val):
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
        
        data['Lowest Low'] = data[low_column].rolling(window=lookback_days, min_periods=1).min()
        data['Highest High'] = data[high_column].rolling(window=lookback_days, min_periods=1).max()

        data['%K'] = ((data[close_column] - data['Lowest Low']) / (data['Highest High'] - data['Lowest Low'])) * 100

        data['%D'] = data['%K'].rolling(window=smoothing_period, min_periods=1).mean()

        data = data.drop(['Lowest Low', 'Highest High'], axis=1)
        
        return data
    
 def kd_IMX_wrapper(kd_val):
    if kd_val < 30 and kd_val >= 0:
        return linear_interpolator(30, .7, 0, 1, rod_val), 0
    elif rod_val <= 70 and rod_val >= 30:
        return 0, 1
    elif rod_val >  70 and kd_val <= 100:
        return linear_interpolator(70, -.7, 100, -1, rod_val), 2
    else: 
        print("Out of Range")
        sys.exit(1)