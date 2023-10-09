import numpy as np
import pandas as pd
from os import path
from sklearn import linear_model
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm, cauchy, nct, genhyperbolic, levy_stable
from math import isnan
import module_parameters



#===============================================================================================================================
#
#                BLOCK FOR FITTING TIME SERIES TO THE ORNSTEIN-UHLENBECK EQUATION
#
#===============================================================================================================================


#-----------------------------------------------------------------------------------------------------------------------

class Spread:

    def __init__(self, input_directory, calculation_mode, list_product_labels_name, label_x, label_y, first_date=None, last_date=None, price_label="log_price_Close_corrected" ):
        '''This reads the time-series of price and its log-returns of the two input labels and stores them in a dataframe.
        The class Spread contains the functions that follow:
        - OLS_regression
        - calc_correlation
        - calc_stationarity
        - calc_spread
        - calc_Ornstein_Uhlenbeck_parameters
        - calc autocorrelation
        '''

        col_corr = price_label.replace("log_","") + "_log_ret" # e.g. "price_Close_corrected_log_ret"

        df_x = pd.read_csv( input_directory + "/Time_series/ret_"+str(label_x)+".csv", usecols=["Date",price_label,col_corr] )
        df_x = df_x.set_index("Date")
        if ( (first_date!=None) and (last_date!=None)): df_x = df_x.loc[first_date:last_date]
        label_xcorr = label_x + "_" + col_corr
        label_xta = label_x+"_"+price_label
        df_x = df_x.rename(columns = {price_label:label_xta, col_corr:label_xcorr})

        df_y = pd.read_csv( input_directory + "/Time_series/ret_" + str(label_y) + ".csv",usecols=["Date", price_label, col_corr])
        df_y = df_y.set_index("Date")
        if ( (first_date!=None) and (last_date!=None)): df_y = df_y.loc[first_date:last_date]
        label_ycorr = label_y + "_" + col_corr
        label_yta = label_y + "_" + price_label
        df_y = df_y.rename(columns = {price_label:label_yta, col_corr:label_ycorr})

        self.time_series = pd.concat([df_x,df_y],axis=1).dropna()
        self.label_x = label_x
        self.label_y = label_y
        self.label_xta = label_xta
        self.label_yta = label_yta
        self.name_col_to_analyse = price_label
        self.name_col_correlation = col_corr
        self.input_dir = input_directory
        self.list_product_labels_name = list_product_labels_name
        self.regr_ret_y_vs_x = None
        self.regr_ret_x_vs_y = None
        self.orn_uhl_regr_y_vs_x_params = None
        self.orn_uhl_regr_x_vs_y_params = None
        self.orn_uhl_regr_y_vs_x_residuals_ts = None
        self.orn_uhl_regr_x_vs_y_residuals_ts = None
        self.orn_uhl_regr_y_vs_x_residuals_params = None
        self.orn_uhl_regr_x_vs_y_residuals_params = None
        self.calculation_mode = calculation_mode
        self.first_date = first_date
        self.last_date  = last_date


    def calc_correlation(self):
        '''Calculates the Pearson's correlation between both time series.'''
        pearson_corr = np.corrcoef(self.time_series[self.label_x + "_" + self.name_col_correlation].to_numpy(),
                                   self.time_series[self.label_y + "_" + self.name_col_correlation].to_numpy())[0, 1]
        return  pd.DataFrame( [[ self.label_x, self.label_y, pearson_corr]] )


    def OLS_regression(self, label_xregr, label_yregr, store_residuals=False):
        '''This calculates the parameters of an ordinary least-squares linear regression between the time-series of both labels.'''
        df_y = pd.DataFrame(self.time_series[label_yregr])

        if ((label_xregr=="Spread_y_vs_x(t-1)")or(label_xregr=="Spread_x_vs_y(t-1)")):
            df_x = pd.DataFrame( self.time_series[label_yregr] )
            df_x=df_x.shift(1)
            df_x = df_x.iloc[1:, :]
            df_y = df_y.iloc[1:, :]
        else:
            df_x = pd.DataFrame( self.time_series[label_xregr]  )

        resu = linear_model.LinearRegression().fit(df_x, df_y)
        resu_slope = resu.coef_[0][0];        # Slope
        resu_intercept = resu.intercept_[0];  # Intercept (y0)
        rsquared = resu.score(df_x, df_y)     # R^2

        if (store_residuals):
            df_aux = df_y - (resu_intercept + df_x * resu_slope)
            if (label_yregr=="Spread_y_vs_x"):
                df_aux.columns = ["residuals_Spread_y_vs_x"]
                self.orn_uhl_regr_y_vs_x_residuals_ts = df_aux.copy()
            elif (label_yregr=="Spread_x_vs_y"):
                df_aux.columns = ["residuals_Spread_x_vs_y"]
                self.orn_uhl_regr_x_vs_y_residuals_ts = df_aux.copy()
            del df_aux
            #exec("self.residuals_orn_uhl_"+label_yregr+" = df_y - ( resu_intercept + df_x * resu_slope )")

        del df_x; del df_y; del resu
        return { 'label':label_yregr+'-vs-'+label_xregr, 'slope':resu_slope, 'intercept':resu_intercept, 'R-squared':rsquared }


    def calc_spread(self):
        '''This calculates the spreads, defined as log(p_A) - gamma · log(p_B), where gamma is the slope of the
        regression of log(p_A(t)/p_A(t-1))-vs-log(p_B(t)/p_B(t-1)). This is calculated for (A=label_x, B=label_y) and vice versa.'''

        label_xregr = self.label_x + "_" + self.name_col_correlation
        label_yregr = self.label_y + "_" + self.name_col_correlation
        self.regr_ret_y_vs_x = self.OLS_regression(label_xregr, label_yregr)
        self.regr_ret_x_vs_y = self.OLS_regression(label_yregr, label_xregr)
        self.time_series["Spread_y_vs_x"] = self.time_series[self.label_yta] - ( self.time_series[self.label_xta] * self.regr_ret_y_vs_x['slope'] )
        self.time_series["Spread_x_vs_y"] = self.time_series[self.label_xta] - ( self.time_series[self.label_yta] * self.regr_ret_x_vs_y['slope'] )
        filepathout = self.input_dir + "/Time_series/Spreads/Spreads_"+self.list_product_labels_name+"/Data/spr_" + str(self.label_x) + "_" + str(self.label_y) + ".csv"
        if (self.calculation_mode=="out_of_sample"):
            suffix = "-oos"
            if ( (self.first_date != None) and (self.last_date != None)): suffix += self.first_date +"_to_"+self.last_date
            filepathout = filepathout.replace(".csv",suffix+".csv")
        self.time_series.to_csv( filepathout )
        if (self.calculation_mode!="out_of_sample"):
            self.stationarity_Spread_y_vs_x = self.calc_stationarity("Spread_y_vs_x")
            self.stationarity_Spread_x_vs_y = self.calc_stationarity("Spread_x_vs_y")


    def calc_stationarity(self, col_name="Spread_y_vs_x", pvalue_thershold=0.05):
        '''This function calculates the stationarity properties using the Augmented Dickey-Fuller test.'''
        my_spread = self.time_series[col_name].to_numpy()
        adfresult = adfuller(my_spread)
        stationary = lambda x: x <= pvalue_thershold
        dict_prod = { "Spread_y_vs_x": self.label_y+"-vs-"+self.label_x, "Spread_x_vs_y": self.label_x+"-vs-"+self.label_y }
        #print("Test statistics=", adfresult[0], "p-value=", adfresult[1])
        return {'products':dict_prod[col_name],'quantity': col_name, 'test_statistics':adfresult[0], 'pvalue':adfresult[1], 'probably_stationary': stationary(adfresult[1]) }


    def calc_Ornstein_Uhlenbeck_parameters(self,demand_stationarity=True):
        '''This functions fits the time-lagged spreads S(t)-vs-S(t-1), and performs an ols linear regression to extract
        the slope and intercept, which provide the parameters of the Ornstein-Uhlenbeck equation. It also calculates the
        parameters of the residuals.'''

        filepath0 = self.input_dir + "/Time_series/Spreads/Spreads_" + self.list_product_labels_name + "/Data/spr_resid_" + self.label_x + "_" + self.label_y + "_"

        if ( (not demand_stationarity) or (self.stationarity_Spread_y_vs_x['probably_stationary'])):
            filepath = filepath0
            filepath += "y_vs_x"
            if (self.calculation_mode=="out_of_sample"):
                filepath += "-oos"
                if ((self.first_date != None) and (self.last_date != None)):
                    filepath += self.first_date + "_to_" + self.last_date
            filepath += ".csv"
            dict_orn_uhl_regr_y_vs_x_params = self.OLS_regression("Spread_y_vs_x(t-1)", "Spread_y_vs_x",True)
            phi      = dict_orn_uhl_regr_y_vs_x_params['slope']
            E0       = dict_orn_uhl_regr_y_vs_x_params['intercept']/(1-phi)
            Rsquared = dict_orn_uhl_regr_y_vs_x_params['R-squared']
            self.orn_uhl_regr_y_vs_x_params = { 'phi':phi, 'E0':E0, 'R-squared':Rsquared }
            self.orn_uhl_regr_y_vs_x_residuals_ts.to_csv(filepath , index=True)
            del dict_orn_uhl_regr_y_vs_x_params; del phi; del E0; del Rsquared;

        if ( (not demand_stationarity) or (self.stationarity_Spread_x_vs_y['probably_stationary'])):
            filepath = filepath0
            filepath += "x_vs_y"
            if (self.calculation_mode == "out_of_sample"):
                filepath += "-oos"
                if ((self.first_date != None) and (self.last_date != None)):
                    filepath += self.first_date + "_to_" + self.last_date
            filepath += ".csv"
            dict_orn_uhl_regr_x_vs_y_params = self.OLS_regression("Spread_x_vs_y(t-1)", "Spread_x_vs_y",True)
            phi = dict_orn_uhl_regr_x_vs_y_params['slope']
            E0 = dict_orn_uhl_regr_x_vs_y_params['intercept'] / (1 - phi)
            Rsquared = dict_orn_uhl_regr_x_vs_y_params['R-squared']
            self.orn_uhl_regr_x_vs_y_params = {'phi': phi, 'E0': E0, 'R-squared': Rsquared}
            self.orn_uhl_regr_x_vs_y_residuals_ts.to_csv( filepath ,index=True)
            del dict_orn_uhl_regr_x_vs_y_params; del phi; del E0; del Rsquared;

        del filepath0;
        return

    def save_Ornstein_Uhlenbeck_parameters(self,input_dir):
        '''This stores to file the parameters of the linear regression of the Ornstein-Uhlenbeck equation.'''
        with open(input_dir+"/Fitting_parameters_OrnUhl_" + self.label_x + "_" + self.label_y +'.csv', 'w') as f:
            f.write('label_x,label_y,regr_type,phi,E0,Rsquared\n')
            if (self.orn_uhl_regr_y_vs_x_params!=None):
                f.write(self.label_x +"," + self.label_y + ",y_vs_x," + "{:.12f}".format(self.orn_uhl_regr_y_vs_x_params['phi']) + "," + "{:.12f}".format(self.orn_uhl_regr_y_vs_x_params['E0']) + "," + "{:.12f}".format(self.orn_uhl_regr_y_vs_x_params['R-squared']) + "\n" )
            if (self.orn_uhl_regr_x_vs_y_params!=None):
                f.write(self.label_x + "," + self.label_y + ",x_vs_y," + "{:.12f}".format(self.orn_uhl_regr_x_vs_y_params['phi']) + "," + "{:.12f}".format(self.orn_uhl_regr_x_vs_y_params['E0']) + "," + "{:.12f}".format(self.orn_uhl_regr_x_vs_y_params['R-squared'])+ "\n")
        f.close()
        del f
        return

    def calc_autocorrelation(self, vec ):
        '''This is the function to calculate autocorrelations. '''
        df_correlation = pd.DataFrame(index=range(9),columns=["autocorrelation"])
        df_correlation.index.names = ['lag']
        for lag in range(9):
            dim = len(vec) - lag
            vec2 = np.roll(vec, -lag)
            vec = vec[:dim]
            vec2 = vec2[:dim]
            df_correlation.loc[lag,"autocorrelation"] = np.corrcoef(vec, vec2)[0, 1]
        print(" The autocorrelations of the spread residuals are:\n\n",df_correlation)
        del vec; del lag; del vec2; del dim

        return df_correlation

    def plot_autocorr(self):
        from module_plots import plot_autocorrelation

        vec=None
        file_stationarity_path = self.input_dir + "/Output" + "/stationarities_" + self.list_product_labels_name + ".csv"
        if not (path.exists(file_stationarity_path)):
            print("\nWARNING: The stationarities file",file_stationarity_path,"does not exist. Therefore the autocorrelations of residuals will not be calculated.\n In order to calculate them, please rerun in mode 'download_and_fit' after the stationarities are calculated.\n")
        df_stationary = pd.read_csv(file_stationarity_path, header=0)
        df_stationary = df_stationary[df_stationary['probably_stationary'] == True]
        df_stationary = df_stationary.set_index( ["products","quantity"] )
        try:
            if (df_stationary.loc[ (self.label_x+"-vs-"+self.label_y,"Spread_x_vs_y"), "probably_stationary" ]):
                vec = self.orn_uhl_regr_x_vs_y_residuals_ts
                df_autocorr = self.calc_autocorrelation( np.array( vec["residuals_Spread_x_vs_y"] ) )
                df_autocorr.to_csv(self.input_dir + "/Time_series/Spreads/Spreads_" + self.list_product_labels_name + "/Data/autocorrelations_" + self.label_x + "_" + self.label_y + "_x_vs_y.csv",index=True)
                plot_autocorrelation(vec,self.input_dir + "/Time_series/Spreads/Spreads_" + self.list_product_labels_name + "/Plots/Autocorrelations/autocorrelation_spread_residuals_" + self.label_x + "_" + self.label_y + "_x_vs_y.pdf", self.label_x + "-vs-" + self.label_y )
        except KeyError:
            pass
        try:
            if (df_stationary.loc[(self.label_y + "-vs-" + self.label_x, "Spread_y_vs_x"), "probably_stationary"]):
                vec = self.orn_uhl_regr_y_vs_x_residuals_ts
                df_autocorr = self.calc_autocorrelation(np.array( vec["residuals_Spread_y_vs_x"] ) )
                df_autocorr.to_csv(self.input_dir + "/Time_series/Spreads/Spreads_" + self.list_product_labels_name + "/Data/autocorrelations_" + self.label_x + "_" + self.label_y + "_y_vs_x.csv",index=True)
                plot_autocorrelation(vec,self.input_dir + "/Time_series/Spreads/Spreads_" + self.list_product_labels_name + "/Plots/Autocorrelations/autocorrelation_spread_residuals_" + self.label_x + "_" + self.label_y + "_y_vs_x.pdf", self.label_y + "-vs-" + self.label_x )
        except KeyError:
            pass
        del vec; del df_stationary; del file_stationarity_path
        return

#-----------------------------------------------------------------------------------------------------------------------

def find_pais_hi_correlation( input_params ):
    '''This function finds the pairs which are suitable for building spreads (filtering by correlation).'''

    mylen = len(input_params.list_product_labels)
    df_corr = pd.DataFrame()
    for i in range(mylen):
        for j in range(i + 1, mylen):
            my_spread = Spread(input_params.input_directory, input_params.calculation_mode, input_params.list_product_labels_name, input_params.list_product_labels[i], input_params.list_product_labels[j]  )
            corr_row = my_spread.calc_correlation()
            df_corr = pd.concat( [df_corr,corr_row],axis=0)
    df_corr.columns=["product_1","product_2","correlation"]
    df_corr = df_corr[ df_corr["correlation"] >= input_params.correlation_threshold ]
    df_corr = df_corr.sort_values(by=['correlation'], ascending=False)
    df_corr = df_corr.reset_index(drop=True)
    df_corr.to_csv( input_params.file_corr_path, index=False )

    del df_corr; del input_params; del corr_row; del mylen; del i; del j; del my_spread

    return

#-----------------------------------------------------------------------------------------------------------------------

def update_stationarity_df( df_inout, my_spread ):
    '''This function stores the results of the stationarity tests of a given spread to a dataframe.'''
    df_aux = pd.DataFrame()
    df_aux[0] = list(my_spread.stationarity_Spread_y_vs_x.values())
    df_aux[1] = list(my_spread.stationarity_Spread_x_vs_y.values())
    df_aux = df_aux.T
    df_aux.rename(columns = { 0:'products',1:'quantity',2:'test-statistcs',3:'pvalue',4:'probably_stationary'}, inplace = True)

    #df_inout = df_inout.append(df_aux)
    df_inout = pd.concat( [df_inout,df_aux],axis=0, ignore_index=True)

    #print(df_inout, "\n")

    del my_spread; del df_aux

    return df_inout

#-----------------------------------------------------------------------------------------------------------------------

def calc_all_spreads( input_params ):
    '''This function calculates all the spreads for the pairs of products with high correlation specified in the
    corresponding correlations file.'''

    print("\n* Now calculating spreads.")

    df0 = pd.read_csv(input_params.file_corr_path, header=0, usecols=["product_1", "product_2"])
    if (df0.empty): raise Exception("\nERROR: The correlations file ",input_params.file_corr_path,"is empty. \n")
    df_stationarity = pd.DataFrame(columns=['products','quantity','test-statistcs','pvalue','probably_stationary'])

    for prod_label1, prod_label2 in zip(df0["product_1"], df0["product_2"]):

        my_spread = Spread(input_params.input_directory, input_params.calculation_mode, input_params.list_product_labels_name, prod_label1, prod_label2 )
        my_spread.calc_spread(  )
        df_stationarity = update_stationarity_df(df_stationarity, my_spread)
        my_spread.calc_Ornstein_Uhlenbeck_parameters()
        my_spread.save_Ornstein_Uhlenbeck_parameters(input_params.ts_directory+"/Spreads/Spreads_"+input_params.list_product_labels_name + "/Data/Fitting_parameters/")
        if (input_params.make_plots):
            df_stationarity.to_csv(input_params.file_stationarity_path, index=False)
            my_spread.plot_autocorr()

    df_stationarity=df_stationarity.sort_values(by=['pvalue']) ; df_stationarity = df_stationarity.reset_index(drop=True)
    df_stationarity.to_csv(input_params.file_stationarity_path,index=False)


    if (len(df0)>1):
        text = "\n* All "+str(len(df0))+" spreads of the "+input_params.list_product_labels_name+" list (whose correlations are stored in\n "+input_params.file_corr_path+") were calculated.\n"
    elif (len(df0)==1):
        text = "\n* The unique spread of the " + input_params.list_product_labels_name + " list (whose correlations are stored in\n " + input_params.file_corr_path + ") was calculated.\n"
    print(text,"The information on stationarity was saved to",input_params.file_stationarity_path,"\n")

    del input_params; del df0; del my_spread; del text

    return

#-----------------------------------------------------------------------------------------------------------------------

def test_stationarity( filepathin  ):
    '''This function analyses the stationarity of time series located in a given '''

    df0 = pd.read_csv( filepathin, header=0, usecols=["field_name"])
    x = df0["field_name"].to_numpy()
    adfresult = adfuller( x )
    print("Test statistics=",adfresult[0],"p-value=",adfresult[1])

    del df0; del x; del adfresult

    return

#-----------------------------------------------------------------------------------------------------------------------


#===============================================================================================================================
#
#                BLOCK FOR FITTING A RANDOM VARIABLE TO PROBABILITY DISTRIBUTIONS
#
#===============================================================================================================================

def fit_to_normal( ts_to_fit ):
    '''This function provides the parameters of the best fitting of the input data to a Gaussian (normal) probability distribution.'''
    normloc   = np.mean(ts_to_fit)
    normsca   = np.std(ts_to_fit)
    loss_norm = - (np.sum(np.log(norm.pdf(ts_to_fit, loc=normloc, scale=normsca ))) / len(ts_to_fit))
    del ts_to_fit
    print("The GLOBAL minimum (normal) is: ",normloc, normsca,"; Loss:",loss_norm )
    return { 'distribution_type':'norm', 'loc_param': normloc, 'scale_param':normsca, 'loss':loss_norm }

# ----------------------------------------------------------------------------------------------------------------------

def read_params_nct( directory, product_label ):
    '''This function reads the parameters of the fitting to an nct probability distribution which are stored in the "directory" (low starting with "product_label").'''

    from glob import glob

    nct_result_file_path = glob(directory+'/spr_fitting_params*nct*')
    if (nct_result_file_path==[]):
        nct_results = {'nct_loc':np.float('NaN')}
    else:
        df0 = pd.read_csv(nct_result_file_path[0],header=0)
        df0 = df0.set_index("spread_name")
        try:
            nct_results = df0.loc[product_label]
        except KeyError:
            nct_results = {'nct_loc': np.float('NaN')}

    del directory; del product_label; del nct_result_file_path; del df0

    return nct_results


# ----------------------------------------------------------------------------------------------------------------------

class FittedTimeSeries:
    '''This class contains the information of the distribution a given dataset (time series) is fitted to.
    It consists of 3 blocks: Initialization (where the data are read), fitting (where the actual fitting is performed)
    and plotting of the fitted distribution (the plots are saved to file).'''

    # Initialization
    def __init__(self, input_params, filename, field_to_read, first_date=None, last_date=None, time_to_maturity=None ) :
        '''This initializes the object with information on the file with raw data to read, as well as the kind of
        fitting which must be performed.'''

        from os import path, makedirs

        # Definition of parameters
        if (input_params.evolution_type == "Ornstein-Uhlenbeck_equation"): # Analysis of spreads
            self.directory_input_data = input_params.ts_directory+"/Spreads/Spreads_"+input_params.list_product_labels_name + "/Data"
        else: # Single product
            self.directory_input_data = input_params.ts_directory
        self.directory_output_data = self.directory_input_data + "/Fitting_parameters"
        self.directory_output_plots = input_params.ts_directory+"/Spreads/Spreads_"+input_params.list_product_labels_name+"/Plots/Fitting_parameters"
        self.is_bond = input_params.is_bond  # If True, we have to convert yields to prices. We do it with assuming zero-coupon, i.e. with Price = 1/(1+Yield)^TTM
        self.efficient_fit_to_levy_stable = input_params.efficient_fit_to_levy_stable
        self.only_plots = input_params.only_plots
        self.filename = filename
        if not (path.exists(self.directory_output_data)): makedirs(self.directory_output_data)
        if not (path.exists(f"{self.directory_output_data}Plots")): makedirs(f"{self.directory_output_data}Plots")
        self.filepathin = f"{self.directory_input_data}/{filename}"
        self.n_random_trials = input_params.n_random_trials
        self.max_n_iter = input_params.max_n_iter
        self.fitting_param_loc = None
        self.fitting_param_scale = None
        self.fitting_param_skew = None
        self.fitting_param_tails1 = None  # This is << df >> for nct function, and << a >> parameter for generalized hyperbolic
        self.fitting_param_tails2 = None  # This is not used for nct function, and << b >> parameter for generalized hyperbolic
        self.consider_p = input_params.consider_p
        self.consider_skewness = input_params.consider_skewness
        self.truncation_limit = -70  # xx: Check if this value is appropriate! For yields, making it 0.7 led to reasonable results. This should ideally depend on the actual product, looking at its lowest value of the past.
        self.alpha_for_ES = 0.025
        self.ttm = time_to_maturity
        self.calculation_mode = input_params.calculation_mode
        self.verbose = input_params.verbose


        # Calculation of returns of PRICE. This stores a df with Date as index, in datetime format, in chronological order. It also stores returns in a sorted np array.
        if not (self.is_bond):
            if ( (first_date==None) and (last_date==None)):
                df_ts = pd.read_csv( self.directory_input_data+"/"+filename, usecols=[field_to_read])
                self.ts_to_fit = np.sort( df_ts[field_to_read].to_numpy() )
            else: # Just out of safety; the time-series should already be in the appropriate time range
                df_ts = pd.read_csv(self.directory_input_data + "/" + filename, usecols=["Date",field_to_read] )
                df_ts = df_ts.set_index("Date")
                df_ts = df_ts.loc[first_date:last_date]
                self.ts_to_fit = np.sort(df_ts[field_to_read].to_numpy())
        else: # is_bond; field_to_read="Price"
            df0 = pd.read_csv(self.filepathin, usecols=["Date",field_to_read])  # IN: In a bond from investing.com, if field_to_read is "Price", then it is indeed a yield (that is what the name of the file states); moreover, we red elsewhere that it means the closing value.
            df0["price_from_yield"] = 100 / ((1 + df0[field_to_read] / 100) ** self.ttm)  # By default, we price the bonds to 100.
            field_to_get_returns = "price_from_yield"
            df0["Date"] = pd.to_datetime(df0["Date"])
            df0.sort_values(by='Date', inplace=True)
            df0.set_index("Date", inplace=True)
            df0["abs_ret"] = df0[field_to_get_returns] - df0[field_to_get_returns].shift(1)
            df0["rel_ret"] = (df0[field_to_get_returns] - df0[field_to_get_returns].shift(1)) / df0[field_to_get_returns]
            df0 = df0.iloc[1:, :]
            num_nans = df0['abs_ret'].isnull().sum()
            if (num_nans != 0): raise Exception("\n ERROR: " + str(num_nans) + " NaNs found. Fix this.\n")
            self.df_ts = df0  # "ts" stands for time-series
            self.ts_to_fit = np.sort(self.df_ts["abs_ret"].to_numpy())

    def change_fitting_params_format(self,fitting_function):
        if (fitting_function == "norm"):
            self.fitting_parameters = { 'distribution_type':'norm', 'mu':self.fitting_parameters['loc_param'],'sigma':self.fitting_parameters['scale_param'], 'loss':self.fitting_parameters['loss']  }
        elif (fitting_function == "nct"):
            self.fitting_parameters = { 'distribution_type':'nct', 'mu':self.fitting_parameters['loc_param'],'sigma':self.fitting_parameters['scale_param'], 'third_param':self.fitting_parameters['skewness_param'], 'fourth_param':self.fitting_parameters['df_param'], 'loss':self.fitting_parameters['loss']  }
        elif (fitting_function == "genhyperbolic"):
            self.fitting_parameters = { 'distribution_type':'genhyperbolic', 'mu':self.fitting_parameters['loc_param'],'sigma':self.fitting_parameters['scale_param'], 'third_param':self.fitting_parameters['b_param'], 'fourth_param':self.fitting_parameters['a_param'] , 'fifth_param':self.fitting_parameters['p_param'], 'loss':self.fitting_parameters['loss']  }
        elif (fitting_function == "levy_stable"):
            self.fitting_parameters = { 'distribution_type':'levy_stable', 'mu':self.fitting_parameters['loc_param'],'sigma':self.fitting_parameters['scale_param'], 'third_param':self.fitting_parameters['beta_param'], 'fourth_param':self.fitting_parameters['alpha_param'], 'loss':self.fitting_parameters['loss']  }


    def check_normality(self, residuals ):
        '''This function performs normality tests of the input variable (a numpy array).'''

        from scipy.stats import anderson, shapiro, kstest
        from math import exp

        shapiro_test = shapiro(residuals)
        pval = shapiro_test.pvalue
        if (pval < 0.05 ): text = "Normality is rejected."
        else:              text = "Failed to reject normality."
        print("\n NORMALITY TESTS:\n Shapiro-Wilk p-value       =", pval,"; "+text)

        # ---

        AD, crit, sig = anderson(residuals, dist='norm')
        # print("Significance Levels:", sig); print("Critical Values:", crit)
        # print("\nA^2 = ", AD)
        AD = AD * (1 + (.75 / 50) + 2.25 / (50 ** 2))
        # print("Adjusted A^2 = ", AD)
        if AD >= .6:
            p = exp(1.2937 - 5.709 * AD - .0186 * (AD ** 2))
        elif AD >= .34:
            p = exp(.9177 - 4.279 * AD - 1.38 * (AD ** 2))
        elif AD > .2:
            p = 1 - exp(-8.318 + 42.796 * AD - 59.938 * (AD ** 2))
        else:
            p = 1 - exp(-13.436 + 101.14 * AD - 223.73 * (AD ** 2))
        if (p < 0.05 ): text = "Normality is rejected."
        else:           text = "Failed to reject normality."
        print(" Anderson-Darling p-value   =", p,"; "+text)

        # -----

        p_val = kstest(residuals, 'norm').pvalue
        if (p_val < 0.05 ): text = "Normality is rejected."
        else:               text = "Failed to reject normality."
        print(" Kolmogorov-Smirnov p-value =", p_val,"; "+text,"\n")
        # "Since the p-value is less than .05, we reject the null hypothesis. We have sufficient evidence to say that the sample data does not come from a normal distribution".

        del residuals; del p; del AD; del pval; del p_val; del text

        return


    # Fitting of the function
    def fit_to_distribution(self, fitting_function):

        if (not (fitting_function in module_parameters.distribution_types)):
            raise Exception("\n ERROR: The function type is " + str(fitting_function) + " while just functions in " + str(module_parameters.distribution_types) + " are allowed. Please, redefine your input data.\n")

        # This fits the data of returns to a given probability distribution.
        if (fitting_function == "nct"):
            print("* Now fitting the data of " + self.filename + " to a non-centered t-student distribution.")
        elif (fitting_function == "genhyperbolic"):
            if (self.consider_p):
                print("* Now fitting the data of " + self.filename + " to a generalized hyperbolic distribution (with p parameter).")
            else:
                print("* Now fitting the data of " + self.filename + " to a generalized hyperbolic distribution (without p parameter).")
        elif (fitting_function == "norm"):
                print("* Now fitting the data of " + self.filename + " to a normal (Gaussian) distribution.")
        elif (fitting_function == "levy_stable"):
            print("\n* Now fitting the data of " + self.filename + " to a stable (i.e. Levy stable) distribution.")

        if ((fitting_function == "norm") ):
            self.fitting_parameters = fit_to_normal( self.ts_to_fit )
            if ((self.calculation_mode in ["fit","download_and_fit"])):
                self.check_normality( self.ts_to_fit  )
        elif (fitting_function == "nct"):
            from module_fitting_tstudent import fit_to_nct_global_minimum
            self.fitting_parameters = fit_to_nct_global_minimum(self.ts_to_fit, self.n_random_trials, self.max_n_iter, self.consider_skewness, self.verbose )
        elif (fitting_function == "genhyperbolic"):
            from module_fitting_genhyperbolic import fit_to_genhyperbolic_global_minimum
            self.fitting_parameters = fit_to_genhyperbolic_global_minimum(self.ts_to_fit,  self.n_random_trials, self.max_n_iter, self.consider_skewness, self.consider_p, self.verbose)
        elif (fitting_function == "levy_stable"):
            from module_fitting_stable import fit_to_stable_global_minimum
            if (self.efficient_fit_to_levy_stable):
                n_random_trials = 1;
                max_n_iter = 12
            else:
                n_random_trials = self.n_random_trials
                max_n_iter      = self.max_n_iter
            self.fitting_parameters = fit_to_stable_global_minimum(self.ts_to_fit, n_random_trials, max_n_iter, self.consider_skewness, self.verbose, read_params_nct( self.directory_output_data, self.filename )  )
        else:
            raise Exception("\nERROR: Unrecognized distribution type ("+str(fitting_function)+").\n")

        if (self.calculation_mode=="out_of_sample"):
            self.change_fitting_params_format(fitting_function)

        return self.fitting_parameters

    # Plotting the fitting
    def plot_fitting(self,plot_several_curves=False, params_several_curves=None):
        if (not self.only_plots):
            from module_plots import plot_histogram
            plot_histogram(self, plot_several_curves, params_several_curves)

# ----------------------------------------------------------------------------------------------------------------------

def truncate_array(dataset_in, truncation_limit):
    ''' This function removes from dataset_in all the values below a threshold (truncation limit).

    :param dataset_in: (numpy array) The array to truncate. It must be sorted.
    :param truncation_limit: (float) Number such that all values below it are removed from the input array.
    :return: truncated dataset_in
    '''

    if (truncation_limit == -np.infty):
        return dataset_in

    i = 0
    while (dataset_in[i] <= truncation_limit):
        i += 1
        if (i == len(dataset_in)):
            raise Exception("\n ERROR: The truncation limit (" + str(truncation_limit) + ") made all the elements of the array to be discarded. Please, check your data.\n")

    return dataset_in[i:]


# ----------------------------------------------------------------------------------------------------------------------
''' 
def calc_ES_discrete(data_in, truncation_limit=-70, alpha=0.025):
    # This function calculates the Expected Shortfall using the EBA formula (see page 18 of
    #Final Draft RTS on the calculation of the stress scenario risk measure under Article 325bk(3) of Regulation (EU)
    #No 575/2013 (Capital Requirements Regulation 2 – CRR2), 17 Dec. 2020. (EBA/RTS/2020/12).
    #Note that due to its mathematical definition, this formula returns positive numbers which correspond to losses.
    #
    #:param data_in: (numpy array of floats) input random variable whose ES will be calculated. IT MUST BE SORTED!
    #:param truncation_limit: (float) Lower truncation limit of the input dataset.
    #:param alpha: (float) lower probability range for the calculation of ES
    #:return: expected_shortfall
    

    my_var = truncate_array(data_in, truncation_limit)

    alphaN = alpha * len(my_var)
    int_alphaN = int(alphaN)
    expected_shortfall = (sum(my_var[0:int_alphaN]) + ((alphaN - int_alphaN) * my_var[int_alphaN])) / (-alphaN)

    return expected_shortfall


# ----------------------------------------------------------------------------------------------------------------------

def calc_ES_continuous(fitted_time_series):
    #This function returns the (numerical) integral of a symbolic function whose parameters are stored in the input
    #parameter (fitted_time_series). Such symbolic function is either a truncated nct or a truncated genhyperbolic function.
    #Note that, again, we define ES to be positive (for losses).
    

    from scipy.stats import nct, genhyperbolic

    # Initialization
    func_type = fitted_time_series.fitting_function
    loc_in = fitted_time_series.fitting_param_loc
    scale_in = fitted_time_series.fitting_param_scale
    skew_in = fitted_time_series.fitting_param_skew
    tail1_in = fitted_time_series.fitting_param_tails1
    tail2_in = fitted_time_series.fitting_param_tails2
    N_points_for_integ = 1000000

    if (func_type == "nct"):
        myvar0 = nct(loc=loc_in, scale=scale_in, nc=skew_in, df=tail1_in)
    elif (func_type == "genhyperbolic"):
        myvar0 = nct(loc=loc_in, scale=scale_in, b=skew_in, a=tail1_in, p=tail2_in)

    # We define the integration limits. Note that we must take into account that part of the probability of was discarded due to the truncation.
    integ_lim_0 = fitted_time_series.truncation_limit
    neo_alpha_trunc = fitted_time_series.alpha_for_ES + myvar0.cdf(fitted_time_series.truncation_limit)
    integ_lim_1 = myvar0.ppf(
        neo_alpha_trunc)  # Remember << myvar =norm( loc=0, scale=1 );  myvar.ppf(0.975) >> gives: 1.95997. << myvar.cdf(-1.96) >> gives 0.024998. Hence ppf is the inverse of cdf. ppf returns a value of "x" (not of probability), this is if loc increases in 10, then the output of ppt also does.
    renorm_factor = 1 / (1 - 2 * myvar0.cdf(
        fitted_time_series.truncation_limit))  # 0.7 xa yield This accounts for the amount of probability which we have discarded through truncation. We assume that it is equally distributed throughout all the non-discarded points.

    # Actual calculation of the (numerical) integral
    x_integ = np.linspace(integ_lim_0, integ_lim_1, N_points_for_integ)
    myvar = myvar0.pdf(x_integ)
    myvar = myvar * x_integ
    integ = sum(myvar) * ((integ_lim_1 - integ_lim_0) / len(myvar))
    integ -= (myvar[0] + myvar[-1]) * ((integ_lim_1 - integ_lim_0) / (2 * len(myvar)))
    integ *= renorm_factor / fitted_time_series.alpha_for_ES
    
    return -integ

    #Old code con ejemplo de prueba q funciona:
    loc_in   = 0#xx fitted_time_series.fitting_param_loc
    scale_in = 0.05#xx fitted_time_series.fitting_param_scale
    skew_in  = 0 #xx  fitted_time_series.fitting_param_skew
    tail1_in = 2.5#xx  fitted_time_series.fitting_param_tails1
    tail2_in = 0 #xx fitted_time_series.fitting_param_tails2
    N_points_for_integ = Nin 

    if (func_type== "nct"):
       myvar0 = nct( loc=loc_in, scale=scale_in, nc=skew_in, df=tail1_in )
    elif (func_type== "genhyperbolic"):
       myvar0 = nct( loc=loc_in, scale=scale_in, b=skew_in, a=tail1_in, p=tail2_in )

    # We define the integration limits. Note that we must take into account that part of the probability of was discarded due to the truncation.
    integ_lim_0 = -0.7 #xx fitted_time_series.truncation_limit
    neo_alpha_trunc = 0.025 + myvar0.cdf( integ_lim_0 )  # xx fitted_time_series.alpha_for_ES + myvar0.cdf( fitted_time_series.truncation_limit)
    integ_lim_1 = myvar0.ppf( neo_alpha_trunc )          # Remember << myvar =norm( loc=0, scale=1 );  myvar.ppf(0.975) >> gives: 1.95997. << myvar.cdf(-1.96) >> gives 0.024998. Hence ppf is the inverse of cdf. ppf returns a value of "x" (not of probability), this is if loc increases in 10, then the output of ppt also does.
    renorm_factor = 1 / (1 - 2 * myvar0.cdf(-0.7))  # xx 0.7!! This accounts for the amount of probability which we have discarded through truncation. We assume that it is equally distributed throughout all the non-discarded points.

    # Actual calculation of the (numerical) integral
    x_integ = np.linspace(integ_lim_0, integ_lim_1, N_points_for_integ)
    myvar = myvar0.pdf( x_integ )
    myvar = myvar * x_integ
    integ = sum(myvar)*( (integ_lim_1 - integ_lim_0)/len(myvar) )
    integ -= ( myvar[0] + myvar[-1] )*( (integ_lim_1 - integ_lim_0)/(2*len(myvar)) )
    integ *= renorm_factor / 0.025 # 

'''




# ------------------------------------------------------------------------------------------------


def print_message_1():
    print("\n====================================================================")
    print("                NOW ANALYSING GOVERNMENT BONDS")
    print("====================================================================")


# ----------------------------------------------------------------------------------------------------------------------

def print_message_2(country_name, n_years):
    print("\n ------------------------------------------------- \n       Now analysing " + str(
        country_name) + ", " + str(n_years) + " years \n -------------------------------------------------")

# ----------------------------------------------------------------------------------------------------------------------

def define_names( input_params, products, quantity ):
    '''This function provides the name of the column to read in the file which contains the random variable to fit to a probability distribution.'''

    prod_label = products.replace("-vs-", "_")  # e.g. BMW.DE-vs-MBG.DE
    col_to_read = quantity
    ts_type = (col_to_read).replace("Spread_", "")  # e.g. "x_vs_y"
    if (ts_type == "y_vs_x"):
        aux0 = prod_label.split("_")[0];
        aux1 = prod_label.split("_")[1]
        prod_label = aux1 + "_" + aux0
        del aux0; del aux1

    if (input_params.evolution_type == "Ornstein-Uhlenbeck_equation"):
        name_col_to_fit = "residuals_Spread_" + ts_type
    else:
        if (input_params.is_bond): name_col_to_fit= "abs_ret"
        else: name_col_to_fit = "rel_ret"
    filename_resid = "spr_resid_" + prod_label + "_" + ts_type + ".csv"

    filename_OrnUhl_params = "Fitting_parameters_OrnUhl_" + filename_resid
    for stri in ["spr_resid_", "_x_vs_y", "_y_vs_x"]:
        filename_OrnUhl_params = filename_OrnUhl_params.replace(stri, "")

    del input_params; del products; del quantity; del col_to_read; del prod_label; del ts_type; del stri

    return filename_resid, name_col_to_fit, filename_OrnUhl_params

# ----------------------------------------------------------------------------------------------------------------------

def create_df_fitting_parameters( input_params, df_stationary ):
    '''This function creates the (empty) dataframe where the parameters of the fitting to residuals will be stored.'''

    list_cols = ["OU_phi", "OU_E0", "OU_Rsquared"]

    if ('norm' in input_params.list_distribution_types):          list_cols += ["normal_loc","normal_scale","normal_loss"]
    if ('nct'  in input_params.list_distribution_types):          list_cols += ["nct_loc", "nct_scale", "nct_skparam", "nct_dfparam", "nct_loss"]
    if ('levy_stable' in input_params.list_distribution_types):   list_cols += ["stable_loc", "stable_scale", "stable_beta_param", "stable_alpha_param", "stable_loss"]
    if ('genhyperbolic' in input_params.list_distribution_types): list_cols += ["ghyp_loc", "ghyp_scale", "ghyp_b_param","ghyp_a_param","ghyp_p_param", "ghyp_loss"]

    if (len(df_stationary)==0): raise Exception("\n ERROR: No stationary time series were found.\n")

    df_fitting_params_index = []
    for i in df_stationary.index:
        filename_resid, trash1, trash2 = define_names(input_params, df_stationary.loc[i,"products"], df_stationary.loc[i,"quantity"])
        df_fitting_params_index.append(filename_resid)

    df_fitting_params = pd.DataFrame( index=df_fitting_params_index, columns=list_cols )
    df_fitting_params = df_fitting_params.rename_axis('spread_name')

    suffix = ''
    for distrib_type in input_params.list_distribution_types: suffix += "_" + str(distrib_type)
    dir_fitting_data = input_params.ts_directory + "/Spreads/Spreads_" + input_params.list_product_labels_name + "/Data/Fitting_parameters/"
    filepath = dir_fitting_data + "spr_fitting_params_" + input_params.list_product_labels_name + suffix + ".csv"
    df_fitting_params.to_csv(filepath,index=True)

    del input_params; del df_stationary; del df_fitting_params; del df_fitting_params_index; del suffix;  del trash1; trash2; del list_cols

    return filepath, dir_fitting_data

# ----------------------------------------------------------------------------------------------------------------------

def update_df_fitting_parameters( file_fit_path, filename_resid, filepath_OrnUhl_params, dict_fitting_parameters):
    '''This function updates the dataframe where the parameters of the fitting to residuals will be stored.'''

    df_fitting_params = pd.read_csv( file_fit_path, header=0 )
    df_fitting_params = df_fitting_params.set_index("spread_name")
    df1 = pd.read_csv(filepath_OrnUhl_params,header=0)
    for i in df1.index:
        df_out_index = "spr_resid_" + str(df1.loc[i,"label_x"]) + "_" + str(df1.loc[i,"label_y"]) + "_" + str(df1.loc[i,"regr_type"]) + ".csv"
        df_fitting_params.loc[df_out_index, "OU_E0"] = str(df1.loc[i, "E0"])
        df_fitting_params.loc[df_out_index, "OU_phi"] = str(df1.loc[i, "phi"])
        df_fitting_params.loc[df_out_index, "OU_Rsquared"] = str(df1.loc[i, "Rsquared"])

    dict_df_fitting   = { 'norm':'normal', 'nct':'nct', 'genhyperbolic':'ghyp', 'levy_stable':'stable' }
    distrib_type_dict = dict_fitting_parameters['distribution_type'] # e.g. nct, norm
    distrib_type_df   = dict_df_fitting[ distrib_type_dict ]         # e.g. nct, normal

    df_fitting_params.loc[filename_resid, distrib_type_df + "_loss"] = dict_fitting_parameters['loss']
    df_fitting_params.loc[filename_resid,distrib_type_df +"_loc"]    = dict_fitting_parameters['loc_param']
    df_fitting_params.loc[filename_resid, distrib_type_df+"_scale"]  = dict_fitting_parameters['scale_param']

    if (distrib_type_dict=="nct"):
        df_fitting_params.loc[filename_resid, distrib_type_df + "_skparam"] = dict_fitting_parameters['skewness_param']
        df_fitting_params.loc[filename_resid, distrib_type_df + "_dfparam"] = dict_fitting_parameters['df_param']
    elif (distrib_type_dict=="levy_stable"):
        df_fitting_params.loc[filename_resid, distrib_type_df + "_alpha_param"] = dict_fitting_parameters['alpha_param']
        df_fitting_params.loc[filename_resid, distrib_type_df + "_beta_param"] = dict_fitting_parameters['beta_param']
    elif (distrib_type_dict=="genhyperbolic"):
        df_fitting_params.loc[filename_resid, distrib_type_df + "_a_param"] = dict_fitting_parameters['a_param']
        df_fitting_params.loc[filename_resid, distrib_type_df + "_b_param"] = dict_fitting_parameters['b_param']
        df_fitting_params.loc[filename_resid, distrib_type_df + "_p_param"] = dict_fitting_parameters['p_param']

    df_fitting_params.to_csv(file_fit_path,index=True)

    del filename_resid; del dict_fitting_parameters; dict_df_fitting; del distrib_type_dict; del distrib_type_df; del df_out_index; del filepath_OrnUhl_params; del df_fitting_params

    return

# ----------------------------------------------------------------------------------------------------------------------
'''
def update_df_fitting_parameters( filename_resid, filepath_OrnUhl_params, df_fitting_params, dict_fitting_parameters):
    #This function updates the dataframe where the parameters of the fitting to residuals will be stored.

    df1 = pd.read_csv(filepath_OrnUhl_params,header=0)
    for i in df1.index:
        df_out_index = "spr_resid_" + str(df1.loc[i,"label_x"]) + "_" + str(df1.loc[i,"label_y"]) + "_" + str(df1.loc[i,"regr_type"]) + ".csv"
        df_fitting_params.loc[df_out_index, "OU_E0"] = str(df1.loc[i, "E0"])
        df_fitting_params.loc[df_out_index, "OU_phi"] = str(df1.loc[i, "phi"])
        df_fitting_params.loc[df_out_index, "OU_Rsquared"] = str(df1.loc[i, "Rsquared"])

    dict_df_fitting   = { 'norm':'normal', 'nct':'nct', 'genhyperbolic':'ghyp', 'levy_stable':'stable' }
    distrib_type_dict = dict_fitting_parameters['distribution_type'] # nct, norm
    distrib_type_df   = dict_df_fitting[ distrib_type_dict ]         # nct, normal

    df_fitting_params.loc[filename_resid, distrib_type_df + "_loss"] = dict_fitting_parameters['loss']
    df_fitting_params.loc[filename_resid,distrib_type_df +"_loc"]    = dict_fitting_parameters['loc_param']
    df_fitting_params.loc[filename_resid, distrib_type_df+"_scale"]  = dict_fitting_parameters['scale_param']

    if (distrib_type_dict=="nct"):
        df_fitting_params.loc[filename_resid, distrib_type_df + "_skparam"] = dict_fitting_parameters['skewness_param']
        df_fitting_params.loc[filename_resid, distrib_type_df + "_dfparam"] = dict_fitting_parameters['df_param']
    elif (distrib_type_dict=="levy_stable"):
        df_fitting_params.loc[filename_resid, distrib_type_df + "_alpha_param"] = dict_fitting_parameters['alpha_param']
        df_fitting_params.loc[filename_resid, distrib_type_df + "_beta_param"] = dict_fitting_parameters['beta_param']
    elif (distrib_type_dict=="genhyperbolic"):
        df_fitting_params.loc[filename_resid, distrib_type_df + "_a_param"] = dict_fitting_parameters['a_param']
        df_fitting_params.loc[filename_resid, distrib_type_df + "_b_param"] = dict_fitting_parameters['b_param']
        df_fitting_params.loc[filename_resid, distrib_type_df + "_p_param"] = dict_fitting_parameters['p_param']

    del filename_resid; del dict_fitting_parameters; dict_df_fitting; del distrib_type_dict; del distrib_type_df; del df_out_index; del filepath_OrnUhl_params

    return df_fitting_params
'''
# ----------------------------------------------------------------------------------------------------------------------

def fit_residuals(input_params):
    '''This function calculates the fitting of the residuals.'''

    df_stationary = pd.read_csv(input_params.file_stationarity_path,header=0)
    df_stationary = df_stationary[ df_stationary['probably_stationary']==True ]
    if not (input_params.only_plots): file_fit_path, dir_fitting_data = create_df_fitting_parameters( input_params, df_stationary )

    if not (input_params.only_plots):
        for i in df_stationary.index:
            filename_resid, name_col_to_fit, filename_OrnUhl_params = define_names(input_params, df_stationary.loc[i,"products"], df_stationary.loc[i,"quantity"])
            mydataset = FittedTimeSeries( input_params, filename_resid, name_col_to_fit )
            for distrib_type in input_params.list_distribution_types:
                fitting_parameters = mydataset.fit_to_distribution( distrib_type )
                update_df_fitting_parameters( file_fit_path, filename_resid, dir_fitting_data+filename_OrnUhl_params, fitting_parameters)
                if ((input_params.make_plots) and (not input_params.only_plots)):
                    mydataset.plot_fitting()
        print(" * Fitting results saved to " + file_fit_path)
        del mydataset; del distrib_type; del i; del file_fit_path

    if (input_params.only_plots):
        from module_plots import plot_histograms_without_fitting, plot_histograms_without_fitting_all_curves
        plot_histograms_without_fitting(input_params)
        plot_histograms_without_fitting_all_curves(input_params)

    #df_fitting_params.to_csv(filepath,index=True)
    # DEV: To do just plotting (without fitting) comment 4 last lines, except <<my_dataset.plot_fitting()>>
    del input_params; del df_stationary;

    return

# ----------------------------------------------------------------------------------------------------------------------

def first_iteration(dataset_in, distrib_type, consider_skewness, loc_param0, sca_param0, skewness_param0, tail_param0, form_param0=None, consider_nonzero_p=False, verbose=0):
    '''This function provides the values to start the Barzilai-Borwein iterations.'''

    grad_form0 = None; form_param1 = None

    if (distrib_type == "nct"):
        from module_fitting_tstudent import calculate_gradient_params
        loss0 = - (np.sum(np.log(nct.pdf(dataset_in, loc=loc_param0, scale=sca_param0, nc=skewness_param0, df=tail_param0)))) / len(dataset_in)
        grad_loc0, grad_sca0, grad_sk0, grad_tail0 = calculate_gradient_params(dataset_in, consider_skewness,loc_param0, sca_param0,skewness_param0, tail_param0)
        factor_accept = 0.95; cauchy_scale = 0.001
    elif (distrib_type == "genhyperbolic"):
        from module_fitting_genhyperbolic import calculate_gradient_params
        from numpy.random import uniform
        loss0 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param0, scale=sca_param0, b=skewness_param0, a=tail_param0,p=form_param0)))) / len(dataset_in)
        count = 0
        while ((isnan(loss0)) and (count<50) ):
            count+=1
            loc_param0 = 0
            sca_param0 = 2 * np.std(dataset_in[int(len(dataset_in) / 4): int(3 * len(dataset_in) / 4)])
            skewness_param0 = 0
            tail_param0 = uniform(1.01,1.99)#1.6
            form_param0 = 0
            loss0 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param0, scale=sca_param0, b=skewness_param0, a=tail_param0, p=form_param0)))) / len(dataset_in)
        if ((isnan(loss0)) and (count==50) ): raise Exception( "\nERROR: Could not find an appropriate starting guess for the generalized hyperbolic function. Please, try manually.\n")
        grad_loc0, grad_sca0, grad_sk0, grad_tail0, grad_form0 = calculate_gradient_params(dataset_in, consider_skewness, consider_nonzero_p, loc_param0, sca_param0, skewness_param0, tail_param0, form_param0 )
        factor_accept = 0.95; cauchy_scale = 0.001;
        for step in [ -10**(-7), 10**(-7), -10**(-8), 10**(-8),-3*10**(-8), 3*10**(-8), -3*10**(-7), 3*10**(-7), -10**(-6), 10**(-6), -10**(-5), 10**(-5) ]:
            loc_param1 = loc_param0 + step * grad_loc0
            sca_param1 = sca_param0 + step * grad_sca0
            skewness_param1 = skewness_param0 + step * grad_sk0
            tail_param1 = tail_param0 + step * grad_tail0
            form_param1 = form_param0 + step * grad_form0
            # Old version: The line below had "levy_stable.pdf"
            loss1 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param1, scale=sca_param1, b=skewness_param1, a=tail_param1, p=form_param1)))) / len(dataset_in)
            if not (isnan(loss1)):break
    elif (distrib_type == "levy_stable"):
        from module_fitting_stable import calculate_gradient_params
        if (verbose > 0): print(" * Now doing the first iteration.")
        loss0 = - (np.sum(np.log(levy_stable.pdf(dataset_in, loc=loc_param0, scale=sca_param0, beta=skewness_param0,alpha=tail_param0)))) / len(dataset_in)
        if (verbose > 1): print("   Now calculating gradient of the first iteration.  ")
        grad_loc0, grad_sca0, grad_sk0, grad_tail0 = calculate_gradient_params(dataset_in, consider_skewness,  loc_param0, sca_param0, skewness_param0, tail_param0 )
        grad_loc0 = np.sign(grad_loc0) * min(500,abs(grad_loc0))
        grad_sca0 = np.sign(grad_sca0) * min(500,abs(grad_sca0))
        if (verbose>1): print("   The gradient is:",grad_loc0, grad_sca0, grad_sk0, grad_tail0)
        factor_accept = 0.95; cauchy_scale = 0.00005
    else:
        raise Exception("\nERROR: Unrecognized function"+distrib_type+"\n")
    skewness_param0 /= 2

    if (isnan(loss0)): # Unable to find an appropriate first step
        if (verbose>0): print("Unable to find an appropriate first step.")
        del dataset_in;del distrib_type;del consider_skewness;
        return loss0, loc_param0,sca_param0,skewness_param0,tail_param0,form_param0,None,None,None,None,None,9999,None,None,None,None,None

    if ( (distrib_type == "levy_stable") or ( (distrib_type == "genhyperbolic") and (isnan(loss1)) )):

        loss_opt = 999; step_opt = 9999
        for step in [ -10**(-8), 10**(-8),-3*10**(-8), 3*10**(-8), -10**(-7), 10**(-7), -3*10**(-7), 3*10**(-7), -10**(-6), 10**(-6), -10**(-5), 10**(-5) ]:
            loc_param1 = loc_param0 + step * grad_loc0
            sca_param1 = sca_param0 + step * grad_sca0
            skewness_param1 = skewness_param0 + step * grad_sk0
            tail_param1 = tail_param0 + step * grad_tail0
            loss1 = - (np.sum(np.log(levy_stable.pdf(dataset_in, loc=loc_param1, scale=sca_param1, beta=skewness_param1,alpha=tail_param1)))) / len(dataset_in)
            if (verbose > 0): print("   Step=", step,"; Loss0=", loss0,  "; Loss1=", loss1)
            if (loss1 < loss_opt):
                loss_opt = loss1
                step_opt = step
        if (loss_opt < factor_accept * loss0):
            loc_param1 = loc_param0 + step_opt * grad_loc0
            sca_param1 = sca_param0 + step_opt * grad_sca0
            skewness_param1 = skewness_param0 + step_opt * grad_sk0
            tail_param1 = tail_param0 + step_opt * grad_tail0
            loss1 = loss_opt

    if ((((distrib_type != "levy_stable") or (loss_opt >= factor_accept * loss0)  )) ):

        if (distrib_type != "genhyperbolic"): loss1=float('NaN') #np.float('NaN')
        count = 0; max_count = 50
        while (count <= max_count) :
            if not (isnan(loss1)):
               if (loss1 < factor_accept * loss0):
                   break
            count += 1
            step = cauchy.rvs(size=1, loc=0, scale=cauchy_scale)[0]
            if (distrib_type == "levy_stable"):
                if ( (count > 12) and (count <= 20)):
                    step = ((-1) ** count) / (10 ** (4 + count / 2 - 6))
                elif ( count in [21,30,40,50,60] ) :
                    max_count = 70
                    cauchy_scale /= 10 # We do this because in some cases one finds huge gradients like: <<The gradient is: -691998447.1385866 -1.1509451688168508 -0.004771834087635612 -0.013121415046846715>>
                    factor_accept *= 0.8
            loc_param1 = loc_param0 + step * grad_loc0
            sca_param1 = sca_param0 + step * grad_sca0
            if (count<=100): skewness_param1 = skewness_param0 + step * grad_sk0
            if (count<=100): tail_param1 = tail_param0 + step * grad_tail0
            if (form_param0!=None): form_param1 = form_param0 + step * grad_form0

            if (distrib_type == "nct"):
                loss1 = - (np.sum(np.log(nct.pdf(dataset_in, loc=loc_param1, scale=sca_param1, nc=skewness_param1, df=tail_param1)))) / len(dataset_in)
            elif (distrib_type == "genhyperbolic"):
                loss1 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param1, scale=sca_param1, b=skewness_param1, a=tail_param1,p=form_param1)))) / len(dataset_in)
            elif (distrib_type == "levy_stable"):
                loss1 = - (np.sum(np.log(levy_stable.pdf(dataset_in, loc=loc_param1, scale=sca_param1, beta=skewness_param1,alpha=tail_param1)))) / len(dataset_in)
            if (verbose > 0): print(count, ") Loss0=", loss0, "; step=", step, "; Loss1=", loss1)

    if ( isnan(loss1)  ): # Unable to find an appropriate first step
        if (verbose>0): print("Loss1=",loss1,"; Loss0=",loss0,"; factor_accept=",factor_accept)
        del dataset_in;del distrib_type;del consider_skewness;
        return loss0, loc_param0,sca_param0,skewness_param0,tail_param0,form_param0,None,None,None,None,None,9999,None,None,None,None,None

    # If the loss of the new point is similar to that of the old point, yet higher, we swap the 0 and 1 points:
    if (loss1 > loss0):
        if (verbose>1): print("Swapping: loss0 ", loss0, "<--> Loss1", loss1)
        aux = loc_param1; loc_param1 = loc_param0; loc_param0 = aux
        aux = sca_param1; sca_param1 = sca_param0; sca_param0 = aux
        aux = skewness_param1; skewness_param1= skewness_param0; skewness_param0 = aux
        aux = tail_param1; tail_param1 = tail_param0; tail_param0 = aux
        aux = form_param1; form_param1 = form_param0; form_param0 = aux
        aux = loss1; loss1=loss0; loss0 = aux
        # Swapping points means that we have to recalculate the gradient at the point 0:
        if ((distrib_type == "nct") or (distrib_type == "levy_stable")):
            grad_loc0, grad_sca0, grad_sk0, grad_tail0 = calculate_gradient_params(dataset_in, consider_skewness,loc_param0, sca_param0,skewness_param0,tail_param0)
        elif (distrib_type == "genhyperbolic"):
            grad_loc0, grad_sca0, grad_sk0, grad_tail0, grad_form0 = calculate_gradient_params(dataset_in, consider_skewness, consider_nonzero_p, loc_param0, sca_param0,skewness_param0,tail_param0, form_param0)

    if (verbose > 0): print("     Parameters of the 1st iteration were found:",loc_param1,sca_param1,skewness_param1,tail_param1,";Loss1=",loss1)

    del dataset_in; del distrib_type; del consider_skewness; del cauchy_scale; del factor_accept; del consider_nonzero_p ; del verbose 

    return loss0, loc_param0,sca_param0,skewness_param0,tail_param0,form_param0, grad_loc0, grad_sca0, grad_sk0, grad_tail0, grad_form0, loss1, loc_param1,sca_param1,skewness_param1,tail_param1,form_param1

# ----------------------------------------------------------------------------------------------------------------------

def first_iteration_single_param( sp, dataset_in, distrib_type, consider_skewness, lim_params, loc_param0, sca_param0, skewness_param0, tail_param0, form_param0=None):
    '''This function provides the values to start the Barzilai-Borwein iterations.'''

    if (distrib_type == "nct"):
        from module_fitting_tstudent import calculate_gradient_params, calculate_gradient_single_param
        loss0 = - (np.sum(np.log(nct.pdf(dataset_in, loc=loc_param0, scale=sca_param0, nc=skewness_param0, df=tail_param0)))) / len(dataset_in)
        my_grad = calculate_gradient_single_param(sp, dataset_in, consider_skewness, lim_params, loc_param0, sca_param0,skewness_param0, tail_param0)
    elif (distrib_type == "genhyperbolic"):
        from module_fitting_genhyperbolic import calculate_gradient_params, calculate_gradient_single_param
        loss0 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param0, scale=sca_param0, b=skewness_param0, a=tail_param0,p=form_param0)))) / len(dataset_in)
        if (isnan(loss0)):
            loc_param0 = 0
            sca_param0 =  2*np.std(dataset_in[int(len(dataset_in) / 4): int(3 *len(dataset_in) / 4)])
            skewness_param0=0
            tail_param0=0
            form_param0=0
            loss0 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param0, scale=sca_param0, b=skewness_param0, a=tail_param0,p=form_param0)))) / len(dataset_in)
            if (isnan(loss0)):
                raise Exception("\nERROR: Could not find an appropriate starting guess for the generalized hyperbolic function. Please, try manually.\n")
        my_grad = calculate_gradient_single_param(sp, dataset_in, consider_skewness, lim_params, loc_param0, sca_param0, skewness_param0, tail_param0)
    elif (distrib_type == "levy_stable"):
        from module_fitting_stable import calculate_gradient_params, calculate_gradient_single_param
        loss0 = - (np.sum(np.log(levy_stable.pdf(dataset_in, loc=loc_param0, scale=sca_param0, beta=skewness_param0,alpha=tail_param0)))) / len(dataset_in)
        my_grad = calculate_gradient_single_param(sp, dataset_in, consider_skewness, lim_params, loc_param0, sca_param0, skewness_param0, tail_param0)

    loss1=999; count = 0
    while (((loss1 > 0.995 * loss0) or (isnan(loss1))) and (count < 50)):

        count += 1
        step = cauchy.rvs(size=1, loc=0, scale=0.0005)[0]

        updp0 = {'a': tail_param0, 'b': skewness_param0}
        updp1 = {'a': tail_param0, 'b': skewness_param0}
        updp1[sp] += step * my_grad

        if (distrib_type == "nct"):
            loss1 = - (np.sum(np.log(nct.pdf(dataset_in, loc=loc_param0, scale=sca_param0, nc=updp1['b'], df=updp1['a'])))) / len(dataset_in)
        elif (distrib_type == "genhyperbolic"):
            loss1 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param0, scale=sca_param0, b=updp1['b'], a=updp1['a'],p=form_param0)))) / len(dataset_in)
        elif (distrib_type == "levy_stable"):
            loss1 = - (np.sum(np.log(levy_stable.pdf(dataset_in, loc=loc_param0, scale=sca_param0, beta=updp1['b'],alpha=updp1['a'])))) / len(dataset_in)

    if (count==50): # Unable to find an appropriate first step
        if not (distrib_type == "levy_stabe"):
            del dataset_in;del distrib_type;del consider_skewness; del loss0; del loss1
            return None,None,None
        else:
            for step in [-10 ** (-8), 10 ** (-8), -3 * 10 ** (-8), 3 * 10 ** (-8), -10 ** (-7), 10 ** (-7), -3 * 10 ** (-7), 3 * 10 ** (-7), -10 ** (-6), 10 ** (-6), -10 ** (-5), 10 ** (-5)]:
                updp0 = {'a': tail_param0, 'b': skewness_param0}
                updp1 = {'a': tail_param0, 'b': skewness_param0}
                updp1[sp] += step * my_grad
                loss1 = - (np.sum(np.log(levy_stable.pdf(dataset_in, loc=loc_param0, scale=sca_param0, beta=updp1['b'],alpha=updp1['a'])))) / len(dataset_in)
                if ((loss1 < 0.98 * loss0) and ( not isnan(loss1))): break
            if( isnan(loss1)):
                return None, None, None

    # If the loss of the new point is similar to that of the old point, yet higher, we swap the 0 and 1 points (including recalculation of the gradient at the point 0):
    if (loss1 > loss0):
        aux = updp1.copy(); updp1 = updp0.copy(); updp0 = aux.copy()
        if ((distrib_type == "nct") or (distrib_type == "levy_stable")):
            my_grad = calculate_gradient_single_param(sp, dataset_in, consider_skewness, lim_params, loc_param0, sca_param0, updp0['b'], updp0['a'] )
        elif (distrib_type == "genhyperbolic"):
            my_grad = calculate_gradient_single_param(sp, dataset_in, consider_skewness, lim_params, loc_param0, sca_param0, updp0['b'], updp0['a'])

    del dataset_in; del distrib_type; del consider_skewness; del loss0; del loss1

    return updp0, my_grad, updp1

# ----------------------------------------------------------------------------------------------------------------------
