'''This module contains the functions necessary for the calculation of the profit out-of-sample.'''


from datetime import datetime
from dateutil.relativedelta import relativedelta
from numpy import log2, exp
import pandas as pd
#pd.set_option('max_columns', 20)
import gc
from module_fitting import Spread, FittedTimeSeries, define_names
from module_parameters import distribution_types_text2
from os import path
from datetime import timedelta
from module_find_data_heatmap import find_thresholds_heatmap

#-----------------------------------------------------------------------------------------------------------------------

def create_df_dates( input_params ):
    '''This function creates the (empty) dataframe where the parameters of the fitting to residuals will be stored.'''

    if not (input_params.evolution_type == "Ornstein-Uhlenbeck_equation"):
        raise Exception("\nERROR: Out-of-sample calculation is just coded for << evolution_type == 'Ornstein-Uhlenbeck_equation' >>.\n")

    # Create dataframe with date intervals
    df_dates = pd.DataFrame(columns=["oos_initial_date","oos_final_date","is_initial_date","is_final_date"]) # "oos" means "out-of-sample"; "is" means "in-sample".

    # Find date intervals
    freq = str(input_params.in_sample_recalibration_freq)
    if ('y' in freq):
        freq = int( freq.replace('y','') )
        reldelkw = "years"
    elif ('m' in freq):
        freq = int( freq.replace('m','') )
        reldelkw = "months"
    elif ('d' in freq):
        freq = int(freq.replace('d', ''))
        reldelkw = "days"
    else:
        raise Exception("\nERROR: Unrecognized format for <<in_sample_recalibration_freq>> ("+str(input_params.in_sample_recalibration_freq)+"). Please, write in format '1y', '12m' or '252d'.\n")
    
    period_is = str(input_params.in_sample_t_interval)
    if ('y' in period_is):
        period_is = int(period_is.replace('y', ''))
        reldelkw2 = "years"
    elif ('m' in period_is):
        period_is = int(period_is.replace('m', ''))
        reldelkw2 = "months"
    elif ('d' in period_is):
        period_is = int(period_is.replace('d', ''))
        reldelkw2 = "days"
    else:
        raise Exception("\nERROR: Unrecognized format for <<in_sample_t_interval> (" + str(input_params.in_sample_t_interval) + "). Please, write in format '1y', '12m' or '252d'.\n")

    list_oos_period_initial_dates = []
    list_oos_period_final_dates   = []
    list_is_period_initial_dates  = []
    list_is_period_final_dates    = []
    first_date  = pd.to_datetime(input_params.first_oos_date, infer_datetime_format=True) #.strftime('%Y-%m-%d')
    last_date   = pd.to_datetime(input_params.last_oos_date,  infer_datetime_format=True) #.strftime('%Y-%m-%d')
    date_oos    = first_date
    while True:
        list_oos_period_initial_dates.append(date_oos)
        if (date_oos > last_date ): break
        if (reldelkw=="years"):      date_oos = date_oos + relativedelta(years=freq)
        elif (reldelkw == "months"): date_oos = date_oos + relativedelta(months=freq)
        elif (reldelkw == "days"):   date_oos = date_oos + relativedelta(days=freq)

    for i in range(len(list_oos_period_initial_dates)-1):
        list_oos_period_final_dates.append( list_oos_period_initial_dates[i+1] - relativedelta(days=1) )
        list_is_period_final_dates.append(  list_oos_period_initial_dates[i]   - relativedelta(days=1) )

    for i in range(len(list_oos_period_initial_dates)-1):
        if (reldelkw2=="years"):      initial_date_is = list_is_period_final_dates[i] -  relativedelta(years=period_is)
        elif (reldelkw2 == "months"): initial_date_is = list_is_period_final_dates[i] -  relativedelta(months=period_is)
        elif (reldelkw2 == "days"):   initial_date_is = list_is_period_final_dates[i] -  relativedelta(days=period_is)
        list_is_period_initial_dates.append( initial_date_is )

    df_dates["oos_initial_date"] = list_oos_period_initial_dates[ : -1]
    df_dates["oos_final_date"]   = list_oos_period_final_dates
    df_dates["is_initial_date"]  = list_is_period_initial_dates
    df_dates["is_final_date"]    = list_is_period_final_dates

    df_dates = df_dates[ df_dates["oos_initial_date"] >= first_date  ]
    df_dates = df_dates[ df_dates["oos_final_date"]   <= last_date]

    del input_params; del list_is_period_initial_dates; del list_is_period_final_dates; del list_oos_period_initial_dates; del list_oos_period_final_dates; del period_is; del freq; del reldelkw; del reldelkw2; del first_date; del last_date

    return df_dates

#-----------------------------------------------------------------------------------------------------------------------

def create_df_results( prod_label_x, prod_label_y, spread_type, df_dates, list_distribution_types, list_max_horizon ):
    '''This function creates the dataframe which will store the fitting information.'''

    assert spread_type in ["y_vs_x","x_vs_y"]
    assert len(list_max_horizon)==1

    list_cols = ["spread_gamma","OU_phi","OU_E0","OU_Rsquared"]
    if ("norm" in list_distribution_types):          list_cols += ["normal_loc","normal_scale","normal_loss"]
    if ("nct"  in list_distribution_types):          list_cols += ["nct_loc","nct_scale","nct_skparam","nct_dfparam","nct_loss"]
    if ("genhyperbolic" in list_distribution_types): list_cols += ["ghyp_loc","ghyp_scale","ghyp_b_param","ghyp_a_param","ghyp_p_param","ghyp_loss"]
    if ("levy_stable" in list_distribution_types):   list_cols += ["stable_loc","stable_scale","stable_beta_param","stable_alpha_param","stable_loss"]
    list_cols += [ "time_horizon", "opt_enter_positive_spread", "opt_pt_positive_spread","opt_sl_positive_spread", "opt_enter_negative_spread", "opt_pt_negative_spread", "opt_sl_negative_spread" ]

    df_results = df_dates.copy()
    df_results["spread_name"] = "spr_resid_"+prod_label_x+"_"+prod_label_y+"_"+spread_type+".csv" # e.g. spr_resid_MS_GS_y_vs_x.csv
    for col_label in list_cols:
        df_results[col_label]=None
    df_results["time_horizon"] = list_max_horizon[0]

    del df_dates; del prod_label_x; del prod_label_y

    return df_results

#-----------------------------------------------------------------------------------------------------------------------

def update_results( i, df_res, type_of_update, info_to_update, spread_type=None ):
    '''This function stores the newly calculated parameters in the dataframe which stores the results.'''

    if (type_of_update=="OU"):

       if (spread_type=="y_vs_x"):
            df_res.loc[i,"OU_phi"] = info_to_update.orn_uhl_regr_y_vs_x_params["phi"]
            df_res.loc[i,"OU_E0" ] = info_to_update.orn_uhl_regr_y_vs_x_params["E0"]
            df_res.loc[i,"OU_Rsquared"] = info_to_update.orn_uhl_regr_y_vs_x_params["R-squared"]
            df_res.loc[i,"spread_gamma"] = info_to_update.regr_ret_y_vs_x['slope']
       elif (spread_type=="x_vs_y"):
            df_res.loc[i,"OU_phi"] = info_to_update.orn_uhl_regr_x_vs_y_params["phi"]
            df_res.loc[i,"OU_E0" ] = info_to_update.orn_uhl_regr_x_vs_y_params["E0"]
            df_res.loc[i,"OU_Rsquared"] = info_to_update.orn_uhl_regr_x_vs_y_params["R-squared"]
            df_res.loc[i, "spread_gamma"] = info_to_update.regr_ret_x_vs_y['slope']

    elif (type_of_update=="fit_residuals"):

        distrib = distribution_types_text2[ info_to_update['distribution_type'] ]
        df_res.loc[i, distrib + '_loc']   = info_to_update['mu']
        df_res.loc[i, distrib + '_scale'] = info_to_update['sigma']
        df_res.loc[i, distrib + '_loss']  = info_to_update['loss']
        if (distrib == 'nct'):
            df_res.loc[i, distrib + '_skparam'] = info_to_update['third_param']
            df_res.loc[i, distrib + '_dfparam'] = info_to_update['fourth_param']
        elif (distrib == 'stable'):
            df_res.loc[i, distrib + '_beta_param'] = info_to_update['third_param']
            df_res.loc[i, distrib + '_alpha_param'] = info_to_update['fourth_param']
        elif (distrib=='ghyp'):
            df_res.loc[i, distrib + '_b_param'] = info_to_update['third_param']
            df_res.loc[i, distrib + '_a_param'] = info_to_update['fourth_param']
            df_res.loc[i, distrib + '_p_param'] = info_to_update['fifth_param']

    elif ( type_of_update=="trading_rules" ):

        for quantity in ["opt_enter_positive_spread", "opt_pt_positive_spread", "opt_sl_positive_spread", "opt_enter_negative_spread", "opt_pt_negative_spread", "opt_sl_negative_spread" ]:
            df_res.loc[i, quantity ] = info_to_update[ quantity ]

    else:

        raise Exception("\nERROR: Unrecognized update type ("+str(type_of_update)+").\n")

    del type_of_update; del info_to_update

    return df_res

#--------------------------------------------------------------------------------------------------------------------

def read_optimal_trading_rules( df_thresholds_one_period, quantity_to_analyse="Sharpe_ratio" ):
    '''This function simply finds the enter-value and profit-taking thresholds which give a maximal value of the quantity_to_analyse.'''

    df_thresholds_one_period = df_thresholds_one_period.reset_index()
    df_thresholds_one_period = pd.DataFrame( df_thresholds_one_period[["enter_value", "profit_taking_param", "stop_loss_param", quantity_to_analyse]] )

    df_positive_en = df_thresholds_one_period[ df_thresholds_one_period["enter_value"] >=0 ]
    df_negative_en = df_thresholds_one_period[ df_thresholds_one_period["enter_value"] < 0]

    if not (df_positive_en.empty):

        if (quantity_to_analyse in ["profit_mean", "Sharpe_ratio", "Sharpe_ratio_with_semideviation"]):
            quant_opt_positive_en = max(df_positive_en[quantity_to_analyse])
        else:
            quant_opt_positive_en = min(df_positive_en[quantity_to_analyse])

        df_positive_en.set_index(quantity_to_analyse, inplace=True)

        df_aux_p = df_positive_en.loc[ quant_opt_positive_en, "enter_value"]
        if (isinstance(df_aux_p, float) or isinstance(df_aux_p, int)):
            spr_positive_enter_value = df_aux_p
        else:
            spr_positive_enter_value = df_aux_p.iat[0]

        df_aux_p = df_positive_en.loc[quant_opt_positive_en, "profit_taking_param"]
        if (isinstance(df_aux_p, float) or isinstance(df_aux_p, int)):
            spr_positive_pt = df_aux_p
        else:
            spr_positive_pt = df_aux_p.iat[0]

        df_aux_p = df_positive_en.loc[quant_opt_positive_en, "stop_loss_param"]
        if (isinstance(df_aux_p, float) or isinstance(df_aux_p, int)):
            spr_positive_sl = df_aux_p
        else:
            spr_positive_sl = df_aux_p.iat[0]

    else:

        spr_positive_enter_value = None
        spr_positive_pt = None

    if not (df_negative_en.empty):

        if (quantity_to_analyse in ["profit_mean", "Sharpe_ratio", "Sharpe_ratio_with_semideviation"]):
            quant_opt_negative_en = max(df_negative_en[quantity_to_analyse])
        else:
            quant_opt_negative_en = min(df_negative_en[quantity_to_analyse])

        df_negative_en.set_index(quantity_to_analyse, inplace=True)

        df_aux_n = df_negative_en.loc[quant_opt_negative_en, "enter_value"]
        if (isinstance(df_aux_n, float) or isinstance(df_aux_n, int)):
            spr_negative_enter_value = df_aux_n
        else:
            spr_negative_enter_value = df_aux_n.iat[0]

        df_aux_n = df_negative_en.loc[quant_opt_negative_en, "profit_taking_param"]
        if (isinstance(df_aux_n, float) or isinstance(df_aux_n, int)):
            spr_negative_pt = df_aux_n
        else:
            spr_negative_pt = df_aux_n.iat[0]

        df_aux_n = df_negative_en.loc[quant_opt_negative_en, "stop_loss_param"]
        if (isinstance(df_aux_n, float) or isinstance(df_aux_n, int)):
            spr_negative_sl = df_aux_n
        else:
            spr_negative_sl = df_aux_n.iat[0]

    else:

        spr_negative_enter_value = None
        spr_negative_pt = None

    opt_tr = { "opt_enter_positive_spread":spr_positive_enter_value, "opt_pt_positive_spread":spr_positive_pt, "opt_sl_positive_spread":spr_positive_sl, "opt_enter_negative_spread":spr_negative_enter_value,"opt_pt_negative_spread": spr_negative_pt ,"opt_sl_negative_spread": spr_negative_sl }

    del df_thresholds_one_period; del quantity_to_analyse; del df_positive_en; del df_negative_en

    return opt_tr

#--------------------------------------------------------------------------------------------------------------------

def read_trading_rules( date_, df_tr, spread_enter_sign, verbose=0 ):
    '''This function reads the trading rules to be applied on that date.'''

    E0=None; gamma=None; en=None; pt=None; sl=None;

    for i in df_tr.index:
        if ( (date_ >= df_tr.loc[i,"oos_initial_date"]) and (date_ <= df_tr.loc[i,"oos_final_date"] ) ):
            E0    = df_tr.loc[i,"OU_E0"]
            gamma = df_tr.loc[i,"spread_gamma"]
            assert(gamma>0)
            en = df_tr.loc[i, "opt_enter_"+spread_enter_sign+"_spread"]  # "ps" stands for "negative spread"
            pt = df_tr.loc[i, "opt_pt_"+spread_enter_sign+"_spread"]
            sl = df_tr.loc[i, "opt_sl_"+spread_enter_sign+"_spread"]
            if (verbose>1): print("-On",str(date_.strftime('%Y-%m-%d')),": E0=","{:.6f}".format(E0),";gamma=","{:.6f}".format(gamma),"; The trading rules are: enter=","{:.6f}".format(en),"; profit-taking=","{:.6f}".format(pt),"; stop-loss=","{:.6f}".format(sl))
            break

    if (E0==None):
        print("\nWARNING: Could not find the trading rules for date ",date_,"; The stored trading rules are:\n",df_tr,"\n")

    del i; del date_; del df_tr; del spread_enter_sign

    return E0, gamma, en, pt, sl

#--------------------------------------------------------------------------------------------------------------------

def oos_profit_measurement( input_params, prod_label_x, prod_label_y, spread_type, filepath_calibration_oos, verbose=0 ):
    '''This function performs the actual measurement of the profitability in units of currency (e.g. dollars) using the
    actually observed time-series of the prices of the products.
    The Spread is defined as log(p^A) - gamma · log(p^B), where gamma is the slope of the regression of the log returns, i.e. log[ p^A(t+1)/p^A(t) ] vs log[ p^B(t+1)/p^B(t) ].
    The information for the calculation is extracted from two files:
    i)  The file which contains the time-series ("ts") prices and spreads (filepath_ts below).
    ii) The file which contains the trading rules ("tr"; filepath_calibration_oos).
    '''

    # Initialization
    if ( spread_type=="y_vs_x" ):
        prod_label_A = prod_label_y
        prod_label_B = prod_label_x
    elif ( spread_type=="x_vs_y" ):
        prod_label_A = prod_label_x
        prod_label_B = prod_label_y
    prod_label_A = str(prod_label_A); prod_label_B = str(prod_label_B)
    my_index = "oos_Spread_"+prod_label_x+"_"+prod_label_y+"_"+spread_type
    filepath_ts = input_params.ts_directory + "/Spreads/Spreads_" + input_params.list_product_labels_name + "/Data/spr_"+prod_label_x+"_"+prod_label_y+".csv" # e.g. Time_series/Spreads/Spreads_banks_USA/Data/spr_BAC_BK.csv
    if not path.exists(filepath_ts): raise Exception("\nERROR: The file "+filepath_ts+"does not exist. Please, make sure that it is written.\n")
    cols_to_read_ts = [ "Date",prod_label_x+"_log_price_Close_corrected", prod_label_y+"_log_price_Close_corrected" ]

    df_ts = pd.read_csv( filepath_ts, header=0, usecols=cols_to_read_ts )

    print(" * The time-series of the spreads were read from", filepath_ts)
    print("   (Its earliest and latest dates are",df_ts.loc[0,"Date"],"and",str(df_ts.loc[len(df_ts)-1,"Date"])+").")

    df_ts[ prod_label_x +"_price_Close_corrected" ] = exp( df_ts[ prod_label_x + "_log_price_Close_corrected"]  )
    df_ts[ prod_label_y + "_price_Close_corrected"] = exp( df_ts[ prod_label_y + "_log_price_Close_corrected"] )
    df_ts["Date"] = pd.to_datetime(df_ts["Date"])
    df_ts = df_ts.set_index("Date")
    df_ts = df_ts.loc[ pd.to_datetime(input_params.first_oos_date):pd.to_datetime(input_params.last_oos_date)]

    if (df_ts.empty):
        raise Exception("\nERROR: The dataframe which contains the time-series (stored in "+filepath_ts+")\n  lacks information in the time-range "+input_params.first_oos_date+" to "+input_params.last_oos_date+". Please, check your data.\n")

    df_profits = pd.DataFrame(index=[my_index],columns=["total_profit_enter_positive_spread","N_enter_positive_spread","avg_price_long_position_enter_positive_spread","avg_cost_building_pair_enter_positive_spread","total_profit_enter_negative_spread","N_enter_negative_spread","avg_price_long_position_enter_negative_spread","avg_cost_building_pair_enter_negative_spread","total_profit","avg_price_long_position","avg_cost_building_pair"])
    df_profits.index.names = ['Spread_name']
    cols_to_read_tr = ["oos_initial_date","oos_final_date","spread_gamma","spread_name","OU_E0","time_horizon","opt_enter_positive_spread","opt_pt_positive_spread","opt_sl_positive_spread","opt_enter_negative_spread","opt_pt_negative_spread","opt_sl_negative_spread"]
    df_tr = pd.read_csv(filepath_calibration_oos,header=0,usecols=cols_to_read_tr)
    df_tr["oos_initial_date"] = pd.to_datetime(df_tr["oos_initial_date"])
    df_tr["oos_final_date"]   = pd.to_datetime(df_tr["oos_final_date"])
    df_tr = df_tr.reset_index()


    # ENTERING AT SPREADS BELOW E0 ("negative"), i.e. calculation of profits with long position in the spread (i.e. entering with spread below its mean E0):

    profit_neg_spread = 0
    invested = False
    avg_enter_plong = 0
    avg_enter_cost = 0
    n_enter = 0
    date_enter = None
    max_horizon =  round(input_params.list_max_horizon[0]*365/252)

    for date_ in df_ts.index:

        E0, gamma, en_ns, pt_ns, sl_ns = read_trading_rules( date_, df_tr, "negative", verbose )
        if (E0==None):continue
        spread = df_ts.loc[date_, prod_label_A + "_log_price_Close_corrected" ] - gamma * df_ts.loc[date_, prod_label_B + "_log_price_Close_corrected" ]
        text1= ""; text2=""
        if (verbose>=2): text1 = str(date_.strftime('%Y-%m-%d'))+ ") Spread="+ str("{:.6f}".format(spread))
        if not (invested):
            spread = df_ts.loc[date_, prod_label_A + "_log_price_Close_corrected"] - gamma * df_ts.loc[date_, prod_label_B + "_log_price_Close_corrected"]
            if ( ( spread < E0 ) and ( spread < E0 + en_ns ) ):
                invested = True
                date_enter = date_
                price_A_en = df_ts.loc[ date_, prod_label_A +"_price_Close_corrected" ] # "en" stands for "enter"
                price_B_en = df_ts.loc[ date_, prod_label_B +"_price_Close_corrected" ]
                if not (input_params.oos_dollar_neutral):
                    gamma_en   = gamma
                else: # We buy one dollar of stock of prod_label_y (ylabel) and sell gamma dollars of stock of prod_label_x (xlabel)
                    gamma_en = price_A_en / price_B_en
                avg_enter_cost += price_A_en - gamma_en * price_B_en
                avg_enter_plong += price_A_en
                n_enter += 1
                if (verbose==1): text1 = str(date_.strftime('%Y-%m-%d'))+ ") Spread="+ str("{:.6f}".format(spread))
                if (verbose>0):  text2 = " ==> Now entering; the price of "+str(prod_label_A)+ " is "+str("{:.2f}".format(price_A_en))+", the price of "+str(prod_label_B)+" is "+str("{:.2f}".format(price_B_en))
        else: # invested
            if ( (spread > E0 + pt_ns ) or (spread < E0 + sl_ns ) or ( date_ >= date_enter + timedelta(days = max_horizon) ) ):
                if ( date_ >= date_enter + timedelta(days = max_horizon) ): print("Exitting due to max horizon.")
                price_A_ex = df_ts.loc[date_, prod_label_A + "_price_Close_corrected"]  # "ex" stands for "exit"
                price_B_ex = df_ts.loc[date_, prod_label_B + "_price_Close_corrected"]
                this_profit = ( price_A_ex - price_A_en ) - gamma_en * ( price_B_ex - price_B_en )
                if ((this_profit>0) or ( date_ >= date_enter + timedelta(days = max_horizon)  ) ):
                    invested = False
                    profit_neg_spread += this_profit
                    if (verbose==1): text1 = str(date_.strftime('%Y-%m-%d'))+ ") Spread="+ str("{:.6f}".format(spread))
                    if (verbose > 0): text2 = "Spread exceeded " +str("{:.6f}".format(E0+pt_ns))+" ("+str("{:.4f}".format(E0))+"+("+str("{:.4f}".format(pt_ns))+")) ==> Now exiting; the price of " + str(prod_label_A) + " is " + str("{:.2f}".format(price_A_ex)) + ", the price of " + str(prod_label_B) + " is " + str("{:.2f}".format(price_B_ex)) + ".\n This profit was: "+str("{:.2f}".format(this_profit)) +" <---\n"
        if ((verbose>0) and ((text1!="")or(text2!="")) ): print( text1, text2)

    if (invested): # We omit the last enter, which we could not close
        avg_enter_plong -= price_A_en
        if not (input_params.oos_dollar_neutral): avg_enter_cost -= ( price_A_en - gamma_en * price_B_en  )
        n_enter -= 1

    if (n_enter !=0):
        avg_enter_plong /= n_enter; avg_enter_cost /= n_enter
        print(" * The total profit if entering for negative spread is:","{:.6f}".format(profit_neg_spread),"\n   The average price of the long position at entering is:","{:.4f}".format(avg_enter_plong),"(",n_enter," enters; profit ratio=","{:.2f}".format(100*profit_neg_spread/avg_enter_plong),"%).")
        if not (input_params.oos_dollar_neutral): print("   The average cost of building the pair is:","{:.4f}".format(avg_enter_cost),"(profit ratio=","{:.2f}".format(100*profit_neg_spread/avg_enter_cost),"%).")
    else:
        print(" * No positions were entered (i.e. the profit is zero).")
    print()


    df_profits.loc[my_index, "total_profit_enter_negative_spread"] = profit_neg_spread
    df_profits.loc[my_index, "N_enter_negative_spread"] = n_enter
    df_profits.loc[my_index, "avg_price_long_position_enter_negative_spread"] = avg_enter_plong
    df_profits.loc[my_index, "avg_cost_building_pair_enter_negative_spread"]  = avg_enter_cost


    #----

    # ENTERING AT SPREADS ABOVE E0 ("positive"), i.e. calculation of profits with short position in the spread (i.e. entering with spread above its mean E0):

    profit_posit_spread = 0
    invested = False
    avg_enter_plong = 0
    avg_enter_cost = 0
    n_enter = 0
    date_enter = None
    max_horizon = round(input_params.list_max_horizon[0] * 365 / 252)

    for date_ in df_ts.index:

        E0, gamma, en_ps, pt_ps, sl_ps = read_trading_rules(date_, df_tr, "positive", verbose)
        if (E0 == None): continue
        spread = df_ts.loc[date_, prod_label_A + "_log_price_Close_corrected"] - gamma * df_ts.loc[date_, prod_label_B + "_log_price_Close_corrected"]
        text1 = ""; text2 = ""
        if (verbose >= 2): text1 = str(date_.strftime('%Y-%m-%d')) + ") Spread=" + str("{:.6f}".format(spread))
        if not (invested):
            spread = df_ts.loc[date_, prod_label_A + "_log_price_Close_corrected"] - gamma * df_ts.loc[date_, prod_label_B + "_log_price_Close_corrected"]
            if ((spread > E0) and (spread > E0 + en_ps)):
                invested = True
                date_enter = date_
                price_A_en = df_ts.loc[date_, prod_label_A + "_price_Close_corrected"]  # "en" stands for "enter"
                price_B_en = df_ts.loc[date_, prod_label_B + "_price_Close_corrected"]
                if not (input_params.oos_dollar_neutral):
                    gamma_en = gamma
                else: # We sell one dollar of stock of prod_label_y (ylabel) and buy gamma dollars of stock of prod_label_x (xlabel)
                    gamma_en = price_A_en / price_B_en
                avg_enter_cost += -price_A_en + gamma_en * price_B_en
                avg_enter_plong += gamma_en * price_B_en
                n_enter += 1
                if (verbose == 1): text1 = str(date_.strftime('%Y-%m-%d')) + ") Spread=" + str("{:.6f}".format(spread))
                if (verbose > 0):  text2 = " ==> Now entering; the price of " + str(prod_label_A) + " is " + str("{:.2f}".format(price_A_en)) + ", the price of " + str(prod_label_B) + " is " + str("{:.2f}".format(price_B_en))
        else:  # invested
            if ((spread < E0 + pt_ps) or (spread > E0 + sl_ps) or (date_ >= date_enter + timedelta(days=max_horizon))):
                if (date_ >= date_enter + timedelta(days=max_horizon)): print("Exitting due to max horizon.")
                price_A_ex = df_ts.loc[date_, prod_label_A + "_price_Close_corrected"]  # "ex" stands for "exit"
                price_B_ex = df_ts.loc[date_, prod_label_B + "_price_Close_corrected"]
                this_profit = -(price_A_ex - price_A_en) + gamma_en * (price_B_ex - price_B_en)
                if ((this_profit > 0) or (date_ >= date_enter + timedelta(days=max_horizon))):
                    invested = False
                    profit_posit_spread += this_profit
                    if (verbose == 1): text1 = str(date_.strftime('%Y-%m-%d')) + ") Spread=" + str("{:.6f}".format(spread))
                    if (verbose > 0):  text2 = "Spread exceeded " + str("{:.6f}".format(E0 + pt_ps)) + " (" + str("{:.4f}".format(E0)) + "+(" + str("{:.6f}".format(pt_ps)) + ")) ==> Now exiting; the price of " + str(prod_label_A) + " is " + str("{:.2f}".format(price_A_ex)) + ", the price of " + str(prod_label_B) + " is " + str( "{:.2f}".format(price_B_ex)) + ".\n This profit was: " + str("{:.2f}".format(this_profit)) + " <---\n"
        if ((verbose > 0) and ((text1 != "") or (text2 != ""))): print(text1, text2)

    if (invested):  # We omit the last enter, which we could not close
        avg_enter_plong -= gamma_en * price_B_en
        avg_enter_cost -= (-price_A_en + gamma_en * price_B_en)
        n_enter -= 1

    if (n_enter != 0):
        avg_enter_plong /= n_enter;
        avg_enter_cost /= n_enter
        print("\n * The total profit if entering for positive spread is:", "{:.6f}".format(profit_posit_spread),
              "\n   The average price of the long position at entering is:", "{:.4f}".format(avg_enter_plong),
              "(",n_enter," enters; profit ratio=", "{:.2f}".format(100 * profit_posit_spread / avg_enter_plong), "%).")
        if not (input_params.oos_dollar_neutral):
            print("   The average cost of building the pair is:", "{:.4f}".format(avg_enter_cost), "(profit ratio=","{:.2f}".format(100 * profit_posit_spread / avg_enter_cost), "%).")
        print()
    else:
        print(" * No positions were entered (i.e. the profit is zero).\n")

    df_profits.loc[my_index, "total_profit_enter_positive_spread"] = profit_posit_spread
    df_profits.loc[my_index, "N_enter_positive_spread"] = n_enter
    df_profits.loc[my_index, "avg_price_long_position_enter_positive_spread"] = avg_enter_plong
    df_profits.loc[my_index, "avg_cost_building_pair_enter_positive_spread"] = avg_enter_cost

    #---

    # Calculation of average results for enter at both positive and negative spreads:
    avg_long_posit = df_profits.loc[my_index, "avg_price_long_position_enter_positive_spread"]
    avg_pair_posit = df_profits.loc[my_index, "avg_cost_building_pair_enter_positive_spread"]
    N_enter_posit  = df_profits.loc[my_index, "N_enter_positive_spread"]
    avg_long_negat = df_profits.loc[my_index, "avg_price_long_position_enter_negative_spread"]
    avg_pair_negat = df_profits.loc[my_index, "avg_cost_building_pair_enter_negative_spread"]
    N_enter_negat  = df_profits.loc[my_index, "N_enter_negative_spread"]
    total_profit   = df_profits.loc[my_index, "total_profit_enter_positive_spread"] + df_profits.loc[my_index, "total_profit_enter_negative_spread"]
    if ((N_enter_posit + N_enter_negat) > 0):
        avg_plong      = ((avg_long_posit*N_enter_posit)+(avg_long_negat*N_enter_negat))/(N_enter_posit+N_enter_negat)
        avg_ppair      = ((avg_pair_posit*N_enter_posit)+(avg_pair_negat*N_enter_negat))/(N_enter_posit+N_enter_negat)
        weighted_profit= (df_profits.loc[my_index, "total_profit_enter_positive_spread"] * df_profits.loc[my_index, "avg_price_long_position_enter_positive_spread"] + df_profits.loc[my_index, "total_profit_enter_negative_spread"] * df_profits.loc[my_index, "avg_price_long_position_enter_negative_spread"])/(df_profits.loc[my_index, "avg_price_long_position_enter_positive_spread"]+df_profits.loc[my_index, "avg_price_long_position_enter_negative_spread"])
        df_profits.loc[my_index, "total_profit"] = total_profit
        df_profits.loc[my_index, "avg_price_long_position"] = avg_plong
        df_profits.loc[my_index, "avg_cost_building_pair"] = avg_ppair

        print(" ** The total profit is:", "{:.6f}".format(total_profit),"\n   The average price of the long position is:", "{:.4f}".format(avg_plong), "(", (N_enter_posit+N_enter_negat), " enters; profit ratio=", "{:.2f}".format(100 * weighted_profit), "%).")
        if not (input_params.oos_dollar_neutral):
            print("   The average cost of building the pair is:", "{:.4f}".format(avg_ppair), ".\n")
    else:
        print("There were no enters, hence there was no profit.\n")
    #---

    filepathout = input_params.output_oos_dir + "/oos_profits_"+ prod_label_x+"_"+prod_label_y+"_"+spread_type+"_" + input_params.list_distribution_types[0] + ".csv"
    df_profits.to_csv( filepathout, index=True )
    print("  The results of the calculation of out-of-sample profits of "+prod_label_x+"_"+prod_label_y+"_"+spread_type+" were saved to "+filepathout )

    del input_params; del prod_label_x; del prod_label_y; del spread_type; del filepath_calibration_oos; del verbose; del df_profits

    return

# --------------------------------------------------------------------------------------------------------------------

def calculate_oos_profits( input_params ):
    '''This function calculates the out-of-sample profits. It consists of three parts:
    i) Fitting of data to the Ornstein-Uhlenbeck equation, with residuals from a given probability distribution;
    ii) Calculation of optimal trading rules;
    iii) Calculation of the out-of-sample profit, with observed prices.  '''

    # INITIALIZATION
    distrib_type = input_params.list_distribution_types[0]
    df_dates    = create_df_dates( input_params )
    prod_label_x = input_params.oos_product_label_x
    prod_label_y = input_params.oos_product_label_y
    spread_type = input_params.oos_spread_type # e.g. "x_vs_y"
    file_suffix = prod_label_x+"_"+prod_label_y+"_"+spread_type+"_"+distrib_type + ".csv"
    if (spread_type=="y_vs_x"): products = input_params.oos_product_label_y + "-vs-" + input_params.oos_product_label_x
    else:                       products = input_params.oos_product_label_x + "-vs-" + input_params.oos_product_label_y
    df_res = create_df_results( prod_label_x, prod_label_y, spread_type, df_dates, input_params.list_distribution_types, input_params.list_max_horizon )
    filename_resid, name_col_to_fit, filename_OrnUhl_params = define_names(input_params, products, "Spread_" + spread_type)
    filepath_calibration_oos = input_params.output_oos_dir + "/oos_calibration_"+file_suffix

    # CALCULATION OF OPTIMAL TRADING RULES

    for i in df_dates.index:

        date_beg = df_dates.loc[i,"is_initial_date"].strftime('%Y-%m-%d');
        date_end = df_dates.loc[i,"is_final_date"].strftime('%Y-%m-%d')

        # Fitting to Ornstein-Uhlenbeck (slope and intercept)
        my_spread = Spread(input_params.input_directory, "out_of_sample", input_params.list_product_labels_name, prod_label_x, prod_label_y, date_beg, date_end )
        my_spread.calc_spread(  )
        my_spread.calc_Ornstein_Uhlenbeck_parameters(False)
        df_res = update_results( i, df_res, "OU", my_spread, spread_type )
        OU_params = {'E0':df_res.loc[i,"OU_E0"], 'tau':-1/log2(df_res.loc[i,"OU_phi"]), 'phi':df_res.loc[i,"OU_phi"]  }                                # Parameters of the Ornstein-Uhlenbeck equation (see eq. (13.2) of Advances in Financial Machine Learning, by Marcos Lopez de Prado, 2018).

        # Fitting of the residuals
        mydataset = FittedTimeSeries( input_params, filename_resid, name_col_to_fit, date_beg, date_end )
        print("\n** Time range between",date_beg,"and",date_end,"**\n")
        rv_params = mydataset.fit_to_distribution( distrib_type )
        df_res = update_results( i, df_res, "fit_residuals", rv_params )

        # Search of optimal trading strategies
        df_thresholds_one_period = find_thresholds_heatmap( input_params, rv_params, OU_params )
        df_thresholds_one_period.to_csv( input_params.output_trad_rules_dir+"/trading_rules_"+file_suffix.replace(".csv",str(date_beg)+"_"+str(date_end)+".csv") )
        opt_tr = read_optimal_trading_rules( df_thresholds_one_period, input_params.quantity_to_analyse )
        df_res = update_results(i, df_res, "trading_rules", opt_tr )

        del my_spread; del mydataset
        gc.collect()

    df_res.to_csv(filepath_calibration_oos,index=False)


    # MEASUREMENT OF THE EARNINGS FROM THE CHOSEN TRADING RULES
    oos_profit_measurement( input_params, prod_label_x, prod_label_y, spread_type, filepath_calibration_oos, input_params.verbose )

    del input_params; del df_dates; del df_res; del opt_tr; del df_thresholds_one_period; del rv_params; del OU_params; del filepath_calibration_oos; del filename_resid; del prod_label_x; del prod_label_y; del file_suffix

    return

#---------------------------------------------------------------------------------------------------------------------------
