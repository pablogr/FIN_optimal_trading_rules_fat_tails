"""

"""

import pandas as pd
from numpy import log
import yfinance as yf
import numpy as np

#-----------------------------------------------------------------------------------------------------

def calculate_returns( product_label, dwl_data, input_params ):
    '''This function receives a dataframe with prices, calculates their returns and stores them to file.
    It includes a column of 'corrected price' which includes dividends assuming that they are reinvested.
    This is, for each ex-dividend date the relative return ('rel_ret_with_dividends') is calculated using
    the non-corrected prices and the dividend. Then the corrected price is calculated as the previous
    corrected price times this relative return.
    '''

    kind_of_price = input_params.kind_of_price

    if ( ("Stock Splits" in list(dwl_data) ) and ( abs(dwl_data["Stock Splits"].sum())>0.00000001)  ):
        my_index = dwl_data["Stock Splits"].max()
        df_aux = pd.DataFrame( dwl_data["Stock Splits"] )
        df_aux = df_aux.reset_index()
        df_aux = df_aux.set_index("Stock Splits"); df_aux.columns = ["Date"]
        print("\n * Split for",product_label,"(",my_index,") on;",df_aux.loc[my_index,"Date"],"\n")
        del df_aux; del my_index

    if (not ("Dividends" in list(dwl_data)) ):
        print("\nSEVERE WARNING: No dividends found for",product_label,"\n")
        dwl_data["Dividends"]=0.0

    my_col   = "price_"+kind_of_price
    my_col_c = my_col + "_corrected"
    my_col_c_log = "log_"+ my_col + "_corrected"
    dwl_data = pd.DataFrame(dwl_data[ [kind_of_price,"Dividends"]])
    dwl_data.columns = [my_col,"Dividends"]

    dwl_data['abs_ret'] = dwl_data[my_col] - dwl_data[my_col].shift(1)
    dwl_data['rel_ret'] = ( dwl_data[my_col] - dwl_data[my_col].shift(1) ) / dwl_data[my_col].shift(1)
    dwl_data['rel_ret_with_dividends'] = ( dwl_data[my_col] + dwl_data["Dividends"] - dwl_data[my_col].shift(1) ) / dwl_data[my_col].shift(1)

    dwl_data[my_col_c] = None
    dwl_data[my_col_c_log] = None
    dwl_data.loc[dwl_data.index[0], my_col_c] = dwl_data.loc[dwl_data.index[0],my_col]

    for i in range(1,len(dwl_data)):
        dwl_data.loc[dwl_data.index[i], my_col_c] = ( dwl_data.loc[dwl_data.index[i-1],my_col_c] ) * ( 1+ dwl_data.loc[dwl_data.index[i],'rel_ret_with_dividends'] )
        dwl_data.loc[dwl_data.index[i], my_col_c_log] = np.log( dwl_data.loc[dwl_data.index[i], my_col_c] )

    # Calculation of returns (abs, rel, log) of the column of interest
    dwl_data[my_col_c+'_abs_ret'] =   dwl_data[my_col_c] - dwl_data[my_col_c].shift(1)
    dwl_data[my_col_c+'_rel_ret'] = ( dwl_data[my_col_c] - dwl_data[my_col_c].shift(1) ) / dwl_data[my_col_c].shift(1)
    dwl_data[my_col_c + '_log_ret'] = dwl_data[my_col_c]  / dwl_data[my_col_c].shift(1)
    for ind in dwl_data.index:
        if ( isinstance(dwl_data.loc[ind,my_col_c + '_log_ret'] , float) or isinstance(dwl_data.loc[ind,my_col_c + '_log_ret'] , int)):
            dwl_data.loc[ind, my_col_c + '_log_ret'] = log( dwl_data.loc[ind,my_col_c + '_log_ret'] )
        else:
            dwl_data.loc[ind, my_col_c + '_log_ret'] = None

    dwl_data.to_csv(input_params.ts_directory + "/ret_" + product_label + ".csv", index=True)

    del product_label; del dwl_data; del input_params; del kind_of_price; del my_col

    return

#-----------------------------------------------------------------------------------------------------

def download_ts( input_params ):
    '''This function downloads time-series from Yahoo Finance and stores them to files.'''

    for product_label in input_params.list_product_labels:
        print("\n\n===================== Now downloading", product_label, "=====================\n")

        # Download
        if ((input_params.first_downloaded_data_date == None) or (input_params.last_downloaded_data_date == None)):
            dwl_data = yf.download(product_label, period=input_params.downloaded_data_period, interval=input_params.downloaded_data_freq,actions=True)
        else:
            dwl_data = yf.download(product_label, start=input_params.first_downloaded_data_date, end=input_params.last_downloaded_data_date,actions=True)

        # Correction to 2 decimal places
        for colname in ["Open","High","Low","Close","Dividends"]:
            if (colname in list(dwl_data)):
                dwl_data[colname] = round(100 * dwl_data[colname]) / 100

        # Calculation of returns and storage
        if not (dwl_data.empty):
            if (dwl_data.index.names == ['Datetime']): dwl_data.index.names = ['Date']
            dwl_data.to_csv( input_params.ts_directory + "/ts_" +product_label+".csv", index=True)
            print(dwl_data)
        else:
            print("\nSEVERE WARNING: Could not download the data of",product_label)
            dwl_data = pd.read_csv(input_params.ts_directory + "/ts_" +product_label+".csv",header=0)
        calculate_returns( product_label, dwl_data, input_params )



''' 
# get ohlcv data for any ticker by period.
data = yf.download("MSFT", period='1mo', interval="5m")

# get ohlcv data for any ticker by start date and end date
data = yf.download("MSFT", start="2017-01-01", end="2020-04-24")

# get intraday data for any ticker by period.
data = yf.download("MSFT", period='1mo', interval="5m")
'''
