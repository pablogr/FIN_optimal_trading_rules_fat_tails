'''
This module contains the functions to read the input files.
'''

import os
from os import getcwd, makedirs
from sys import path
import pandas as pd
import numpy as np
from numpy import log

path.append(getcwd())
import input         # This imports a file called input.py, which stores the user-defined input parameters, and is located
                     # at the directory where Python is called. For example, you can open a terminal at the my_running_dir
                     # directory and do: $ python path_to_source_code/main.py . This will make the code work.
from module_parameters import dict_price_types, distribution_types, distribution_types_text

#-----------------------------------------------------------------------------------------------------------------------

class InputParams:
    '''This is a data class which contains the values of the parameters determined by the user for the calculations.'''

    def __init__(self):

        # Directory where the file with user-defined input parameters input.py is stored
        self.input_directory = getcwd()

        self.print_intial_message()
        
        
        #===================================================================================================================
        #                                 BLOCK OF DOWNLOAD AND FITTING DATA
        #===================================================================================================================

        # Directory to store the data
        self.ts_directory = self.input_directory + "/Time_series"

        # Directory to store the data
        self.output_directory     = self.input_directory + "/Output";
        if not (os.path.exists(self.output_directory)): makedirs(self.output_directory)
        #xxx self.output_downl_fit_dir = self.output_directory + "/Output_fitting"
        self.output_trad_rules_dir= self.output_directory + "/Output_trading_rules"
        self.output_oos_dir       = self.output_directory + "/Output_out-of-sample"

        # Kind of calculations to be performed. Set to either "download", "fit", "download_and_fit" or "out_of_sample". "download" downloads the data from Yahoo Finance; "fit" fits the data to probability distributions (and eventually to the Ornstein-Uhlenbeck equation), "download_and_fit" makes both; "out_of_sample" calculates the profitability in specified periods.
        try: self.calculation_mode = input.calculation_mode
        except AttributeError: self.calculation_mode = "download_and_fit"
        assert (self.calculation_mode in ["download","fit", "find_trading_rules","download_and_fit","download_fit_and_find_trading_rules", "out_of_sample"])
        if (self.calculation_mode in ["download","download_and_fit","download_fit_and_find_trading_rules"]):
            self.download_data = True # True if you want data --time-series-- to be downloaded (from Yahoo Finance).
        else:
            self.download_data = False
        if (self.calculation_mode in ["fit","download_and_fit","download_fit_and_find_trading_rules"]):
            self.fit_data = True  #  True if you want data to fit the data to given distributions; otherwise False.
        else:
            self.fit_data = False

        # The variable below defines how the Spread is defined. If it is "y_vs_x" then the Spread is log(Price_{label_y}) - gamma·log(Price_{label_x}), where gamma is the slope of the Return_{label_y}-vs-Return_{label_x}. If it is "x_vs_y", then the prices and returns are swapped.
        self.oos_spread_type = input.oos_spread_type
        assert self.oos_spread_type in ["x_vs_y", "y_vs_x"]

        # Labels of the products whose prices will take part in the out-of-sample calculation
        self.oos_product_label_x = input.oos_product_label_x
        self.oos_product_label_y = input.oos_product_label_y

        # Type of the probability distribution to fit the random variables. It must be either 'norm', 'nct', 'genhyperbolic' or 'levy_stable'.
        try:
            list_aux = input.list_distribution_types
            self.list_distribution_types = ['norm', 'nct','genhyperbolic','levy_stable', 'johnsonsu' ]
            for distrib in ['norm', 'nct', 'genhyperbolic','levy_stable','johnsonsu']: # We order the distributions
                if not (distrib in list_aux):
                    self.list_distribution_types.remove(distrib)
        except AttributeError:
            self.list_distribution_types = [ "norm" ]

        try:  self.discount_rate = input.discount_rate
        except AttributeError: self.discount_rate = None
        if (self.discount_rate != None): assert self.discount_rate >= 0

        try: self.transaction_costs_rate = input.transaction_costs_rate
        except AttributeError: self.transaction_costs_rate = None
        if (self.transaction_costs_rate!=None): assert self.transaction_costs_rate>=0

        # Number of times that each set of starting parameters in the fitting of the random variable is tried. Each trial corresponds to a different first iteration of the gradient descent algorithm.
        try: self.n_random_trials = input.n_random_trials
        except AttributeError: self.n_random_trials = 200

        try: self.consider_skewness = input.consider_skewness
        except AttributeError: self.consider_skewness = True

        try: self.consider_p = input.consider_p
        except AttributeError: self.consider_p = True

        # Max number of iterations in each search for optimal parameters (using the gradient descent algorithm, with a given starting value for the distribution parameters).
        try: self.max_n_iter= input.max_n_iter
        except AttributeError: self.max_n_iter = 100  # If the optimal solution was not reached in 100 iterations, probably it will not be improved in further 100 iterations.

        try: self.efficient_fit_to_levy_stable = input.efficient_fit_to_levy_stable
        except AttributeError: self.efficient_fit_to_levy_stable = True

        # Period of the data to download
        try: self.downloaded_data_period = input.downloaded_data_period
        except AttributeError: self.downloaded_data_period = None #'12y'

        # Frequency of the data to download
        try: self.downloaded_data_freq = input.downloaded_data_freq
        except AttributeError: self.downloaded_data_freq = '1d'

        # First date for our analysis (if unset, then it is the first date of the downloaded period); write in format "YYYY-MM-DD".
        try: self.first_downloaded_data_date = input.first_downloaded_data_date
        except AttributeError:  self.first_downloaded_data_date = None

        # Last date for our analysis (if unset, then it is the last date of the downloaded period); write in format "YYYY-MM-DD".
        try: self.last_downloaded_data_date = input.last_downloaded_data_date
        except AttributeError: self.last_downloaded_data_date = None

        if (len(input.list_product_labels) > 2):
            self.list_risk_factor_labels = input.list_product_labels[2]
        else:
            raise Exception( "\nERROR: Please, include a list of risk factors in the variable list_product_labels in input.in.\n")

        # Defining the time series of the risk-free rate
        if (len(input.list_product_labels) < 4):
            print("\nWARNING: The risk-free rate will be assumed to be zero.\n")
            self.risk_free_rate = pd.DataFrame(
                index=pd.date_range(start=self.first_downloaded_data_date, end=self.last_downloaded_data_date, freq='B',tz=None))
            self.risk_free_rate["risk_free_rate"] = 0
        else:
            risk_free_filename = str(input.list_product_labels[3])
            df_Rf = pd.read_csv(os.path.join(self.ts_directory, risk_free_filename + ".csv"))
            df_Rf["Date"] = pd.to_datetime(df_Rf["Date"])
            df_Rf = df_Rf.sort_values('Date', ascending=True)
            df_Rf.set_index("Date", inplace=True)
            df_Rf["risk_free_rate"] = log(1 + df_Rf["Yield"] / 100)
            if (self.first_downloaded_data_date != None):
                df_Rf = df_Rf.loc[self.first_downloaded_data_date:]
            if (self.last_downloaded_data_date != None):
                df_Rf = df_Rf.loc[:self.last_downloaded_data_date]
            self.risk_free_rate = pd.DataFrame(df_Rf["risk_free_rate"].copy())
            del df_Rf;
            del risk_free_filename

        # Labels of the products whose time-series must be downloaded.
        if (self.calculation_mode == "out_of_sample"):
            self.list_product_labels_name = input.list_product_labels_name  # Name of the list of products; it must be the one specified in <<list_product_labels>> when the download was performed.
            self.list_product_labels = [input.oos_product_label_x, input.oos_product_label_y]
        else:
            self.list_index_labels        = input.list_index_labels
            self.list_product_labels_name = input.list_product_labels[0]
            self.list_product_labels      = input.list_product_labels[1]


        self.file_corr_path = self.output_directory + "/correlations_" + self.list_product_labels_name + ".csv"
        self.file_stationarity_path = self.output_directory + "/stationarities_" + self.list_product_labels_name + ".csv"

        try:
            self.is_bond = input.is_bond
            print("* The products to analyse will be treated as bonds, this is the input data will be taken as yields which will be converted to prices.\n")
        except AttributeError:
            self.is_bond = False

        try:
            self.evolution_type = input.evolution_type
        except AttributeError:
            self.evolution_type = "Ornstein-Uhlenbeck_equation"
        if (self.evolution_type == None): self.evolution_type = "Ornstein-Uhlenbeck_equation"
        assert self.evolution_type in ["Ornstein-Uhlenbeck_equation","single_product"]
        if (self.evolution_type == "Ornstein-Uhlenbeck_equation"):
            self.field_to_read="Spread"
        else:
            if (self.is_bond): self.field_to_read="Price" # For bonds from finanzen.net.
            else: self.field_to_read="Close" # For stocks from Yahoo Finance.


        # Kind of price to be used
        try: self.kind_of_price = input.kind_of_price
        except AttributeError: self.kind_of_price = None
        assert self.kind_of_price in [ None, "Open", "High", "Low", "Close" ]
        if (self.kind_of_price == None): self.kind_of_price = "Close"

        try:
            self.verbose = input.verbose
        except AttributeError:
            self.verbose = 0
        if (self.verbose==None): self.verbose=0


        print("\nNow working with the list of "+str(self.list_product_labels_name)+", which contains",len(self.list_product_labels)," products with labels:\n",self.list_product_labels)

        if (self.download_data):
            print("\n===========================================================================================")
            print("                          Now downloading time-series")
            print("===========================================================================================\n")
            print("The identifiers of the time-series to download are:\n", self.list_product_labels)
            if (self.list_index_labels!=[]): print("\nThe indices to download are:\n", self.list_index_labels)

            if ( ((self.first_downloaded_data_date == None) or (self.last_downloaded_data_date == None)) and (self.downloaded_data_period==None) ):
                self.downloaded_data_period = "12y"
            print("\nWe download data with frequency "+self.downloaded_data_freq+".")
            if ((self.first_downloaded_data_date != None) and (self.last_downloaded_data_date != None)):
                print("The first and last dates that we will use are:",self.first_downloaded_data_date,", ",self.last_downloaded_data_date)
            else:
                print("The downloaded data corresponds to the last "+self.downloaded_data_period,".")
            print("We calculate the returns of " + dict_price_types[self.kind_of_price], "prices.")
            print("The downloaded data are stored to", self.ts_directory)


        # Creation of directories to store data:

        #if ( self.calculation_mode in ["download", "fit", "download_and_fit","download_fit_and_find_trading_rules"]):
        #    if not (os.path.exists(self.output_downl_fit_dir)): makedirs(self.output_downl_fit_dir)
        if (self.calculation_mode in ["find_trading_rules","download_fit_and_find_trading_rules" ]):
            for my_dir in [self.output_trad_rules_dir, self.output_trad_rules_dir+"/Plots", self.output_trad_rules_dir+"/Results"]:
                if not (os.path.exists(my_dir)): makedirs(my_dir)
        if ( self.calculation_mode == "out_of_sample" ):
            if not (os.path.exists(self.output_oos_dir)): makedirs(self.output_oos_dir)


        list_directories = [self.ts_directory, self.ts_directory+"/Spreads", self.ts_directory+"/Spreads/gammas",
                            self.ts_directory+"/Plots", self.ts_directory+"/Spreads/Spreads_"+self.list_product_labels_name ,
                            self.ts_directory+"/Spreads/Spreads_"+self.list_product_labels_name + "/Data",
                            self.ts_directory + "/Spreads/Spreads_" + self.list_product_labels_name + "/Data/Fitting_parameters",
                            self.ts_directory+"/Spreads/Spreads_"+self.list_product_labels_name+"/Plots",
                            self.ts_directory + "/Spreads/Spreads_" + self.list_product_labels_name + "/Plots/Autocorrelations",
                            self.ts_directory+"/Spreads/Spreads_"+self.list_product_labels_name+"/Plots/Fitting_parameters"]


        try:
            self.check_convergence_trading_rules = input.check_convergence_trading_rules
        except AttributeError:
            self.check_convergence_trading_rules = False
        if (self.check_convergence_trading_rules):
            self.dir_trading_rules_convergence = self.output_trad_rules_dir + "/Trading_rules_convergence"
            if not (os.path.exists(self.dir_trading_rules_convergence)): makedirs(self.dir_trading_rules_convergence)

        for my_directory in list_directories:
            if not (os.path.exists(my_directory)): makedirs(my_directory)

        file_name = self.ts_directory+"/Spreads/gammas/_README.txt"
        content = "Files stored in this directory contain the gammas, this is the slopes of the clouds of points\n" \
                  "of the COMMON TRENDS of the returns of two stocks or cryptocurrencies.\n" \
                  "The code which calculates this can be found in module_fitting.py:\n" \
                  '  slope_y_vs_x = linear_model.LinearRegression().fit( pd.DataFrame(df_ct["common_trend_x"]) ,pd.DataFrame(df_ct["common_trend_y"]) ).coef_[0][0]'+"\n"+'  slope_x_vs_y = linear_model.LinearRegression().fit(pd.DataFrame(df_ct["common_trend_y"]),pd.DataFrame(df_ct["common_trend_x"])).coef_[0][0]'
        with open(file_name, 'w') as file:
            file.write(content)
        del file_name; del content

        if ((self.fit_data) or (self.download_data)):
            print("\n===========================================================================================")
            print("                            Now fitting data ")
            print("===========================================================================================\n")

            # Lower limit to the correlation to discard a pair in an analysis for spread
            try:
                self.correlation_threshold = input.correlation_threshold
            except AttributeError:
                self.correlation_threshold = 0.7
            if (self.correlation_threshold ==None): self.correlation_threshold = -1.1
            if (self.correlation_threshold < -1):
                print("There is no correlation threshold imposed in the calculation of spreads.\n")
            else:
                print("The correlation lower threshold (below whose value a pair of products is ignored to define a Spread) is "+str(self.correlation_threshold)+".\n")

        try: self.verbose       = input.verbose           # If set to 1 or 2 intermediate results of the calculations are printed to screen.
        except AttributeError: self.verbose = 0

        try: self.make_plots       = input.make_plots    # Set to False if plots are not to be made in the execution.
        except AttributeError: self.make_plots = True

        try: self.only_plots       = input.only_plots    # Set to True to just make the plots, without any calculation.
        except AttributeError: self.only_plots = False



        #===================================================================================================================
        #                                 BLOCK OF TRADING RULES
        #===================================================================================================================

        if not (self.calculation_mode in ["find_trading_rules","download_fit_and_find_trading_rules", "out_of_sample"]): return

        self.evolution_type   = input.evolution_type   # This variable indicates which type of calculations will be performed. Possibilities: "Ornstein-Uhlenbeck_equation" or "single_product"
        try:
            self.output_type  = input.output_type    # Kind of quantity which is calculated. Possibilities: "heat-map" or "optimal_solution"
        except AttributeError:
            self.output_type = "heat-map"

        if (self.output_type == "heat-map"):text0 = ""
        else:text0 = "the optimal "
        print("\n===========================================================================================")
        print("           Now calculating " + text0 + "profit-taking and stop-loss thresholds")
        print("===========================================================================================\n")

        # Path to the file which contains the parameters of the Ornstein-Uhlenbeck equation and its residuals. Set to 'default' to use the data calculated in previous steps. This is also used in the "out_of_sample" calculations.
        try:
           self.path_rv_params = input.path_rv_params
        except AttributeError:
           self.path_rv_params = None
        if (self.path_rv_params == 'default'):

            self.path_rv_params = self.ts_directory + "/Spreads/Spreads_" + self.list_product_labels_name + "/Data/Fitting_parameters/spr_fitting_params_"+self.list_product_labels_name
            suffix=""
            for distr in self.list_distribution_types: suffix += "_"+distr
            self.path_rv_params += suffix+".csv"
            if not (os.path.exists(self.path_rv_params)):
                print(self.path_rv_params, "NOT found")
                self.path_rv_params = self.ts_directory + "/Spreads/Spreads_" + self.list_product_labels_name + "/Data/Fitting_parameters/spr_fitting_params_"+self.list_product_labels_name + "_norm_nct_genhyperbolic_levy_stable_johnsonsu.csv"
            if not (os.path.exists(self.path_rv_params)):
                raise Exception("\n ERROR: The specified path to store the parameters of the random variables ("+str(self.path_rv_params)+")\n does not exist. Please, provide it.\n")

        if (self.path_rv_params!=None):

            from pandas import read_csv

            my_path = self.path_rv_params
            if ( os.path.exists(my_path) ):
                self.df_rv_params = read_csv(self.path_rv_params, header=0)
                self.list_product_labels = self.df_rv_params["spread_name"].values.tolist()
                self.df_rv_params = self.df_rv_params.set_index("spread_name")
            else: # We merge the different files where the parameters are stored
                list_possible_names = ["_norm.csv", "_nct.csv", "_levy_stable.csv", "_genhyperbolic.csv"]
                for i in range(6):
                    for stri in list_possible_names:
                        self.path_rv_params = self.path_rv_params.replace(stri,".csv")
                list_rv_params_files = []
                for distrib in input.list_distribution_types:
                    list_rv_params_files.append( self.path_rv_params.replace(".csv","_"+distrib+".csv") )
                count_present = 0
                for i in range(len(list_rv_params_files)):
                    file_rv_path = list_rv_params_files[i]
                    if (os.path.exists(file_rv_path)):
                        count_present += 1
                        df_aux2 = pd.read_csv(file_rv_path)
                        self.list_product_labels = df_aux2["spread_name"].values.tolist()
                        df_aux2 = df_aux2.set_index("spread_name")
                        if (count_present==1):
                            df_aux1 = df_aux2.copy()
                        if (count_present>1):
                            df_aux2 = df_aux2.drop(["OU_phi", "OU_E0", "OU_Rsquared"],axis=1)
                            if (df_aux2.index != df_aux1.index ):
                                raise Exception("\n ERROR: the 'spread_name' column of the files "+str(list_rv_params_files)+"\n differs. Please, make sure that they are equal so that they can be merged.\n")
                            #df_aux2 = df_aux2.drop["spread_name"]
                            df_aux1 = pd.concat([df_aux1,df_aux2],axis=1)
                suffix = ""
                for distrib in input.list_distribution_types:
                    suffix += "_"+distrib
                self.path_rv_params = self.path_rv_params.replace(".csv",suffix+".csv")
                df_aux1.to_csv(self.path_rv_params,index=True)
                self.df_rv_params = df_aux1.copy()
                del df_aux2; del df_aux1; del file_rv_path

            print("* We will analyse the products that follow:",self.list_product_labels,"\n  (read from ",self.path_rv_params,").")

        else: # self.path_rv_params not specified

            try:
                self.list_E0        = input.list_E0             # List of mean reversion parameters to analyse
            except AttributeError:
                if (self.evolution_type=="Ornstein-Uhlenbeck_equation"):
                    raise Exception("\nERROR: Since you specified Ornstein-Uhlenbeck equation in input.py, you have to provide 'list_E0'.\n")
                else:
                    self.list_E0 = [None]
    
            try:
                self.list_tau      = input.list_tau              # List of half-life parameters to analyse (this determines the speed parameter phi)
            except AttributeError:
                if (self.evolution_type == "Ornstein-Uhlenbeck_equation"):
                    raise Exception("\nERROR: Since you specified Ornstein-Uhlenbeck equation in input.py, you have to provide 'list_tau'.\n")
                else:
                    self.list_tau = [None]

            self.list_mu            = input.list_mu            # List of means of the residuals (or rv's) to analyse
            self.list_sigma         = input.list_sigma         # List of standard deviations of the residuals (or rv's) to analyse
            try: self.list_3rdparam = input.list_3rdparam      # List of 3rd params of the distribution to try (optional)
            except AttributeError: self.list_3rdparam = [None]
            try: self.list_4thparam = input.list_4thparam      # List of 4th params of the distribution to try (optional)
            except AttributeError: self.list_4thparam = [None]
            try: self.list_5thparam = input.list_5thparam      # List of 5th params of the distribution to try (optional)
            except AttributeError: self.list_5thparam = [None]

        try:
            self.poisson_probability = float(max(min( float(input.poisson_probability), 1), 0))  # Probability of a Poisson event. Set to None for no Poisson events.
            if (self.poisson_probability == 0): self.poisson_probability = None
        except AttributeError:
            self.poisson_probability = None
        try:
            self.new_value_after_poisson_event = input.new_value_after_poisson_event  # Value acquired after a Poisson event. If evolution_type is "single_product" this is the value of the price itself after the event. If evolution_type is "Ornstein-Uhlenbeck" then it is the new long-term mean (E0).
        except AttributeError:
            self.new_value_after_poisson_event = None
        try:
            self.new_value_after_poisson_event_increase = abs(input.new_value_after_poisson_event_increase)
        except AttributeError:
            self.new_value_after_poisson_event_increase = None
        try:
            self.new_value_after_poisson_event_decrease = -abs(input.new_value_after_poisson_event_decrease)
        except AttributeError:
            self.new_value_after_poisson_event_decrease = None

        try:                                               # Method for calculating the optimal solution (if <<output_type="optimal solution">>). Either "Barzilai-Borwein" or "bisection"
            self.method_optimal_solution=input.method_optimal_solution
        except AttributeError:
            self.method_optimal_solution="bisection"

        # Method for the calculation of the profit. Must be in ["enter_ensured", "enter_random_one_deal", "enter_random_many_deals"].
        # "enter_ensured" is the method presented in Chapter 13 of Marcos Lopez de Prado's "Advances in Financial machine learning". In it, the position is 100% sure built in t=0, and after reaching a threshold the process starts again.
        # "enter_random_one_deal" means that the initial value of the spread is its average (for Ornstein-Uhlenbeck), then it randomly varies until the prescribed enter level is reached; after reaching a threshold the process starts again.
        # "enter_random_many_deals" means that the initial value of the spread is its average (for Ornstein-Uhlenbeck), then it randomly varies until the prescribed enter level is reached; after this, the spread continues to evolve, and we can build new positions, whose return is added to make a cumulative return.
        try:  # Method for calculating the optimal solution (if <<output_type="optimal solution">>). Either "Barzilai-Borwein" or "bisection"
            self.method_for_calculation_of_profits = input.method_for_calculation_of_profits
        except AttributeError:
            if (self.evolution_type == "Ornstein-Uhlenbeck_equation"):
                self.method_for_calculation_of_profits = "enter_random_many_deals"
            else: self.method_for_calculation_of_profits = "enter_ensured"
        if (self.evolution_type == "single_product"):
                self.method_for_calculation_of_profits = "enter_ensured"


        # Set to True if heatmaps for each enter_value are to be plotted
        try: self.plot_heatmaps_for_individual_enter_value = input.plot_heatmaps_for_individual_enter_value
        except AttributeError: self.plot_heatmaps_for_individual_enter_value = False

        if (self.evolution_type == "Ornstein-Uhlenbeck_equation"):
            list_aux = input.list_enter_value # List of considered values to enter the position
        self.list_enter_value = []
        for en in list_aux:
            if (abs(en)>0.0000000000001): self.list_enter_value.append(en)
            else: self.list_enter_value.append( 0.000000000000001 * np.sign(np.mean( np.array(list_aux))) )

        # For plotting heatmaps we check that there is only one sign of enter values
        if (self.calculation_mode=="find_trading_rules"):
            check_ev = self.list_enter_value[0] * self.list_enter_value[-1]
            if ((abs(check_ev)>0.00000001) and (check_ev<0) ):
                raise Exception("\nERROR: In a calculation in mode 'find_trading_rules' all elements of list_enter_value must have the same sign.\nPlease, rewrite it in input.in and rerun your calculation.\n ")

        self.list_profit_taking = input.list_profit_taking # List of profit-taking thresholds toa analyse when plotting the heat-maps (2D plots to present the performance of the trading strategy)
        self.list_stop_loss     = input.list_stop_loss     # List of stop-loss thresholds toa analyse when plotting the heat-maps
        self.list_max_horizon   = input.list_max_horizon   # Number of periods (discrete times) to exit the position (investment) if no threshold was reached before
        self.strategy_duration  = input.strategy_duration  # Number of days that the strategy is expected to be used; in practice this means that its return is evaluated in one period with this duration (in each iteration of the Monte Carlo algorithm)..
        self.min_num_trials     = input.min_num_trials     # Minimum number of calculations to evaluate the heat-map (set to a high value, e.g. 100000, and make sure that your results are converged in this quantity; such convergence is partly automatically required)

        self.input_directory    = input.input_directory    # Directory where the file with the user-defined input parameters (input.py) lies
        self.output_directory   = input.input_directory+'/Output' # Directory where the outputs are stored
        self.quantity_to_analyse= input.quantity_to_analyse # Quantity that will be plotted in the heatmaps, or which will be optimized in "optimal_solution" calculations.

        assert self.evolution_type  in ["Ornstein-Uhlenbeck_equation", "single_product"] # The first option indicates that the given variable follows the Ornstein-Uhlenbeck equation, which is commonly used for pairs trading; the second possibility indicates that the evolution of a single stock (which is not assumed to follow the Ornstein-Uhlenbeck equation) is simulated.
        assert self.output_type       in ["heat-map", "optimal_solution"]                # The first option plots different values for a heat map (2D chessboard plot with different colors); the second option indicates that the optimal solution is sought.
        assert self.quantity_to_analyse in ["Sharpe_ratio", "average", "standard_deviation", "semideviation", "Sharpe_ratio_with_semideviation", "VaR", "ES", "probab_loss"]
        assert self.method_optimal_solution in ["Barzilai-Borwein", "bisection"]
        assert self.method_for_calculation_of_profits in ["enter_ensured", "enter_random_one_deal", "enter_random_many_deals"]

        for my_directory in [ self.output_directory, self.output_trad_rules_dir+"/Results", self.output_trad_rules_dir+"/Plots"]:
            if not ( os.path.exists( my_directory )): makedirs( my_directory )

        print("The used input parameters are stored in " + str(self.input_directory))
        print("The outputs will be stored in " + self.output_directory, "\n")
        if (self.evolution_type == "Ornstein-Uhlenbeck_equation"):
            print("The calculations will be based on the Ornstein-Uhlenbeck equation with the following kinds of distributions of the residuals: "+str(self.list_distribution_types) )
            if (self.path_rv_params == None):
                print("The list of mean reversion parameters analysed at the Ornstein-Uhlenbeck equation is:    ",self.list_E0)
                print("The list of half-life parameters analysed at the Ornstein-Uhlenbeck equation is:         ", self.list_tau)
            rv_kind = "residuals"
        else:
            rv_kind = "random variable"

        if (self.path_rv_params == None):
            print("\nThe list of location parameters of the " + rv_kind + " is: ",self.list_mu)
            print("The list of scaling parameters of the "+rv_kind+" is: ", self.list_sigma )
            if (self.distribution_type=="t-student"):
                print("The list of " + rv_kind + " skewness parameter values of the involved random variable is: ",self.list_3rdparam)
                print("The list of " + rv_kind + " degrees of freedom values of the involved random variable is: ",self.list_4thparam)
            elif (self.distribution_type=="stable"):
                print("The list of " + rv_kind + " skewness parameter values of the involved random variable is: ",self.list_3rdparam)
                print("The list of " + rv_kind + " tail parameter values of the involved random variable is: ",self.list_4thparam)
            elif (self.distribution_type=="hyperbolic"):
                print("The list of " + rv_kind + " 3rd parameters analysed of the involved random variable is: ",self.list_3rdparam)
                print("The list of " + rv_kind + " 4th parameters analysed of the involved random variable is: ",self.list_4thparam)
                print("The list of " + rv_kind + " 5th parameters analysed of the involved random variable is: ",self.list_5thparam)
        if (self.poisson_probability==None):
            print("No Poisson events are included in the model variable.")
        else:
            textpoisson={"single_product":"price of the product","Ornstein-Uhlenbeck_equation":"mean of the Ornstein-Uhlenbeck equation"}
            print("We will consider Poisson events with probability ",self.poisson_probability, "per unit time (e.g. daily)." )
            if (self.new_value_after_poisson_event_increase!=None):
                print("They will increase the E0 (mean of the Ornstein-Uhlenbeck process) by",100*self.new_value_after_poisson_event_increase,"%")
                print("and decrease it by",100 * self.new_value_after_poisson_event_decrease, "%, both with probability 50%.")

            elif (self.new_value_after_poisson_event != None):
                print( "They set the \n",textpoisson[self.evolution_type], "to ",self.new_value_after_poisson_event+"." )

        if ((self.transaction_costs_rate==None) or (abs(self.transaction_costs_rate)<0.000000001)):
            print("\nNo transaction costs will be considered.")
        else:
            print("\nThe rate of transaction costs (yearly rate measured in units of currency) will be "+str(100*self.transaction_costs_rate)+" %.")
            print("This corresponds to a yearly rate of transaction costs for the spread c="+str(-np.log(1-self.transaction_costs_rate))+".")

        if ((self.discount_rate==None) or (abs(self.discount_rate)<0.000000001)): print("No discount factors will be considered.")
        else: print("The (yearly) rate used to calculate discount factors is "+str(100*self.discount_rate)+" %." )

        print("\nThe chosen method for the calculation of profits is <<"+self.method_for_calculation_of_profits+">>. This means that")
        if (self.method_for_calculation_of_profits=="enter_ensured"):
            print("in each iteration of the Monte Carlo process the position is built in t=0 (with probability equal to 1), and")
            print("after reaching a threshold (profit-taking, stop-loss or time-horizon) the process (a new iteration) starts again.")
            print("(This is the method presented in Chapter 13 of Marcos Lopez de Prado's 'Advances in Financial Machine Learning').\n")
        elif  (self.method_for_calculation_of_profits=="enter_random_one_deal"):
            print("the initial value of the spread is the expected value of the Ornstein-Uhlenbeck process; then the spread randomly varies")
            print("until a threshold (profit-taking, stop-loss or time-horizon) is reached; after reaching a threshold the process starts again.\n")
        elif  (self.method_for_calculation_of_profits=="enter_random_many_deals"):
            print("the initial value of the spread is the expected value of the Ornstein-Uhlenbeck process; then the spread randomly varies")
            print("until a threshold (profit-taking, stop-loss or time-horizon) is reached; after this, the spread continues to evolve, and ")
            print("we can build new positions, whose return is added. The total profit of each iteration is therefore a cumulative return.\n")

        if (self.evolution_type == "Ornstein-Uhlenbeck_equation"):
            print("The",len(self.list_enter_value),"analysed values to enter the position are between", "{:.5f}".format(self.list_enter_value[0]),"and", str( "{:.5f}".format((self.list_enter_value[-1])))+"." )
        if (abs(self.list_profit_taking[0])>0.00001): pt1="{:.5f}".format(self.list_profit_taking[0])
        else: pt1 = self.list_profit_taking[0]
        if (abs(self.list_profit_taking[-1])>0.00001): pt2 ="{:.5f}".format(self.list_profit_taking[-1])
        else: pt2 = self.list_profit_taking[-1]
        print("The",len(self.list_profit_taking),"analysed profit-taking parameters (to add to the enter value) lie between "+str(pt1)+" and "+str(pt2)+".")
        if (len(self.list_stop_loss)>1): print("The",len(self.list_stop_loss),"analysed stop-loss parameters (to add to the enter value) lie between", self.list_stop_loss[0], "and",self.list_stop_loss[-1])
        else: print("The only considered stop-loss parameter will be "+str(self.list_stop_loss[0])+".")
        if (len(self.list_max_horizon) > 1): print("The",len(self.list_max_horizon),"analysed parameters of maximum horizon till closing the position is between",min(self.list_max_horizon),"and",max(self.list_max_horizon))
        else: print("The only considered maximum horizon till closing the position will be "+str(self.list_max_horizon[0]))
        print("time units (a time unit corresponds to the time difference between two rows in the input time series datafile).")
        print("The number of days that the strategy is expected to be used is "+str(self.strategy_duration )+";\nthis is the maximum time simulated in each iteration of the Monte Carlo algorithm used to find optimal rules.\n")

        if (self.output_type=="heat-map"):
            if not (self.only_plots): print("The goal is to plot heat-maps. They will be elaborated using at least",self.min_num_trials,"calculations for each point.")
        else:
            print("The goal is to find the optimal thresholds which optimize the "+self.quantity_to_analyse+". Each pair of thresholds\nwill be evaluated using at least",self.min_num_trials,"calculations for each point.")


        #===================================================================================================================
        #                                 BLOCK OF OUT-OF-SAMPLE CALCULATIONS
        #===================================================================================================================

        if (self.calculation_mode=="out_of_sample"):

            print("\n===========================================================================================")
            print("                   Now performing out-of-sample profitability calculations ")
            print("===========================================================================================\n")

            if ( len(self.list_distribution_types)>1 ): raise Exception("\nERROR: The calculation mode is << out_of_sample >> but the list of distributions to analyse consists of "+str(len(self.list_distribution_types))+" elements.\n("+str(self.list_distribution_types)+"). Please, include a single distribution in it to run out-of-sample calculations.\n")
            if (len(self.list_max_horizon)>1):          raise Exception("\nERROR: The calculation mode is << out_of_sample >> but the list of horizons to analyse consists of "+str(len(self.list_max_horizon))+" elements.\n("+str(self.list_max_horizon)+"). Please, include a single maximum horizon in it to run out-of-sample calculations.\n")

            try:
                self.first_oos_date = input.first_oos_date   # First date for the out-of-sample calculation. E.g. '2016-11-30'
            except AttributeError:
                from datetime import datetime
                self.first_oos_date = datetime.today().strftime('%Y-%m-%d')
            try:
                self.last_oos_date = input.last_oos_date   # Last date for the out-of-sample calculation. E.g. '2022-11-30'
            except AttributeError:
                from datetime import datetime
                from dateutil.relativedelta import relativedelta
                self.last_oos_date = self.first_oos_date - relativedelta(years=1)
            try:
                self.in_sample_t_interval = input.in_sample_t_interval   # Time interval prior to the out-of-sample period which forms the in-sample period (i.e. whose information is used to calibrate the random variables). Set e.g. to '100d' for 100 trading days or '2y' for two years.
            except AttributeError:
                self.in_sample_t_interval = '4y'
            try:
                self.in_sample_recalibration_freq = input.in_sample_recalibration_freq   # Frequency for the recalibration of the parameters of the input variable (i.e. Ornstein-Uhlenbeck parameters and parameters of the random variable corresponding to its residuals).
            except AttributeError:
                self.in_sample_recalibration_freq = '1y'
            try:
                self.oos_dollar_neutral = input.oos_dollar_neutral
            except AttributeError:
                self.oos_dollar_neutral = False
            if (self.oos_dollar_neutral):
                print(" * In the out-of-sample calculations we will use dollar-neutral portfolios, this is we will buy (or sell) one unit of the product A\n    and sell (or buy) gamma' units of the product B, being gamma' the quotient between both prices as the time of entering.\n")
            else:
                print(" * In the out-of-sample calculations we will NOT use dollar-neutral portfolios; we will buy (or sell) one unit of the product A\n   and sell (or buy) gamma units of the product B, where gamma is the slope of the regression between log-returns of A vs B.\n")

            # The variable below defines how the Spread is defined. If it is "y_vs_x" then the Spread is log(Price_{label_y}) - gamma·log(Price_{label_x}), where gamma is the slope of the Return_{label_y}-vs-Return_{label_x}. If it is "x_vs_y", then the prices and returns are swapped.
            self.oos_spread_type = input.oos_spread_type
            assert self.oos_spread_type in ["x_vs_y", "y_vs_x"]

            # Labels of the products whose prices will take part in the out-of-sample calculation
            self.oos_product_label_x = input.oos_product_label_x
            self.oos_product_label_y = input.oos_product_label_y

            # Name of the subdirectory where the spreads are stored.
            try:
                self.list_product_labels_name = input.list_product_labels_name # It must be the same as the first element of the list_product_labels input parameter which is used when << calculation_mode >> is "download".
            except AttributeError:
                self.list_product_labels_name = self.list_product_labels[0]


    def print_intial_message(self):

        print("\n      ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒╗")
        print("      ▒▒                                                                                  ▒▒║")
        print("      ▒▒                                                                                  ▒▒║")
        print('      ▒▒          ██████╗   ██╗   ██╗   ██████╗    ██╗   ██╗   ██████╗  ████████╗         ▒▒║')
        print('      ▒▒         ██╔═══██╗  ██║   ██║  ██╔════╝    ██║   ██║  ██╔════╝  ╚══██╔══╝         ▒▒║')
        print('      ▒▒         ████████║  ██║   ██║  ██║  ████╗  ██║   ██║   ██████═╗    ██║            ▒▒║')
        print('      ▒▒         ██╔═══██║  ██║   ██║  ██║   ██╔╝  ██║   ██║   ╚════██║    ██║            ▒▒║')
        print('      ▒▒         ██║   ██║   ██████╔╝   ██████╔╝    ██████╔╝   ██████╔╝    ██║            ▒▒║')
        print('      ▒▒         ╚═╝   ╚═╝   ╚═════╝    ╚═════╝     ╚═════╝    ╚═════╝     ╚═╝            ▒▒║')
        print("      ▒▒                                                                                  ▒▒║")
        print("      ▒▒   THE PROGRAM FOR THE ANALYSIS OF TRADING RULES USING FAT-TAILED DISTRIBUTIONS   ▒▒║")
        print("      ▒▒                            (P. Risueño, 2023-2025)                               ▒▒║")
        print("      ▒▒                                                                                  ▒▒║")
        print("      ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒║")
        print("       ╚════════════════════════════════════════════════════════════════════════════════════╝\n")

        print("\n                                           See references:\n")
        print("                                  ● M. López de Prado, 'Advances in")
        print("                            financial Machine Learning', Chap. 13 (2018);")
        print("                          ● A. Göncü, E. Akyildirim, 'A stochastic model for   ")
        print("                        commodity pairs trading', Quantitative Finance (2016); ")
        print("                   /█\    ● Pablo Risueño et al., 'The effect of fat tails on   /█\                                 ")
        print("                   ███      rules for optimal pairs trading', SSRN, (2023).     ███                                 ")
        print("                  /███\                                                        /███\                                ")
        print("                  |███|                                                        |███|                                ")
        print("                  |███|                                                        |███|                                ")
        print("                  █████                                                        █████                                ")
        print("                 |█████|                                                      |█████|                               ")
        print("                |███████|                                                    |███████|                              ")
        print("               |█████████|                                                  |█████████|                             ")
        print("              |███████████|                                                |███████████|                            ")
        print("             /█████████████\                                              /█████████████\                           ")
        print("___,,,,▄▄▄▄▄█████████████████▄▄▄▄▄,,,,____                 _____,,,,▄▄▄▄▄█████████████████▄▄▄▄▄,,,,____ ")
        print("▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀^^--    --^^▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀\n\n")



