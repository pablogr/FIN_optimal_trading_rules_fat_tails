# This is the file which contains the parameters defined by the user for the analysis of optimal trading strategies through the
# seek for profit-taking and stop-loss thresholds. See the corresponding theory in Chapter 13 of Marcos Lopez de Prado's "Advances in financial ML".

from os import getcwd
from numpy import array, linspace, arange, log


# Directory where the file with user-defined input parameters input.py is stored
input_directory  = getcwd() 


# ====================================================================================================
#              GENERAL INPUT PARAMETERS (for both blocks)
# ====================================================================================================

# Kind of calculations to be performed. Set to either "download", "fit", "download_and_fit", "find_trading_rules" or "out_of_sample". "download" downloads the data from Yahoo Finance; "fit" fits the data to probability distributions (and eventually to the Ornstein-Uhlenbeck equation), "download_and_fit" makes both; "find_trading_rules" calculates the optimal trading rules; "out_of_sample" calculates the profitability in specified periods.
calculation_mode =  "out_of_sample" #"find_trading_rules"#"download_and_fit"#

# The variable below indicates the type of trading that will be performed. Possibilities: "Ornstein-Uhlenbeck_equation" (for pairs trading) or "single_product" (for single-product trading).
evolution_type = "Ornstein-Uhlenbeck_equation" 

# Types of the probability distributions to fit the random variables. It must be either 'norm', 'nct', 'genhyperbolic' or 'levy_stable'.
list_distribution_types = ['norm']#['norm','nct','genhyperbolic','levy_stable']  


# Labels of the products whose time-series must be downloaded. Note that they are taken into account only if "download" is in calculation_mode; otherwise the starting point of the calculations is the stationarity file (in the Output directory).
#list_product_labels =  [ "oil_companies", ["SHEL", "BP"]]#[ "XOM", "CVX", "COP", "PSX", "VLO", "MPC", "MRO", "EOG", "BP", "SHEL", "TTE", "E", "EQNR", "CNQ", "OXY", "NTGY.MC" ] ] #, "BZ=F"  # Oil companies (worldwide) sold in NY; MPC and PSX are available just from 2011 and 2012, respectively.  "BZ=F" is the oil barrel today, I think.
#list_product_labels =  [ "oil_companies", ["BP", "SHEL" ] ]  
#list_product_labels =  [ "oil_companies", ["CVX", "TTE" ] ]  
#list_product_labels = ["oil_companies",["PSX","EOG"]]
#list_product_labels = [ "car_manufacturers", [ "BMW.DE","MBG.DE" ]]
#list_product_labels = ["cola_companies",["KO","PEP"]]
#list_product_labels = ["retailers_USA",["BBY","TGT"]]
#list_product_labels = ["banks_USA",["GS","MS"]]#  ["banks_USA",["GS","C","BAC","MS","JPM","BLK","BX","WFC","USB","TFC","PNC","COF","BK"]]
list_product_labels = ["credit_cards",["V","MA"]]
#list_product_labels = [ "cryptocurrencies", ["BTC-USD","ETH-USD"]]
# OTHERS:
#list_product_labels = [ "electric_power_companies", ["EOAN.DE", "RWE.DE"]]#[ "electric_power_companies", ["ENGI.PA", "D2G.DE", "ENEL.MI", "IBE.MC", "ELE.MC", "RED.MC", "NTGY.MC", "EOAN.DE", "RWE.DE", "EBK.DE", "SSE.L", "EDP.LS" ] ]# European electricity companies. Note that SSE is in pounds! 
#list_product_labels = [ "car_manufacturers", ["RNO.PA","STLA.PA","VOW3.DE","VOLV-A.ST","VOLV-B.ST","BMW.DE","MBG.DE","NISA.F","TL0.DE"] ] # Car corporations sold in Europe ()
#list_product_labels = [ "cryptocurrencies","SOL-USD","ADA-USD","XRP-USD","DOT-USD","DOGE-USD","XMR-USD","AVAX-USD","TRX-USD"] ]
#list_product_labels = ["Grifols",["GRF.MC","GRF-P.MC"]]
#list_product_labels = ["retailers_USA",["WMT","WBA","COST","HD","KR","BBY","TGT"]]
#list_product_labels = ["nasdaq_composite",["T","TMUS","TXN","IBM","AVGO","CSCO","ADBE"]]
#list_product_labels = ["computer_hardware",["AMD","NVDA","DELL","INTC","TXN","IBM"]]
#list_product_labels = [ "FAANG",["GOOG","AAPL","NFLX","META","AMZN","MSFT","TSLA"]]
#list_product_labels = [ "car_manufacturers", ["BMW.DE","MBG.DE","RNO.PA","STLA.PA"] ] 


# Set to more than 0 (i.e. 1 or 2) to print partial results from screen
verbose = 2

make_plots = False
only_plots = False



# ====================================================================================================
#       BLOCK FOR DOWNLOADING DATA AND FITTING THEM TO PROBABILITY DISTRIBUTIONS
# ====================================================================================================

# This is the file which contains the parameters defined by the user for the 
# downloading financial time series and fitting them to appropriate distributions.

# First date for our analysis (if unset, then it is the first date of the downloaded period); write in format "YYYY-MM-DD" or define as None.
first_downloaded_data_date = '2015-10-15' # '2010-11-30'    # cards: '2015-10-15' ; crypto: '2021-05-02'; retailers: '2017-03-15';  rest:'2010-11-30' 

# Last date for our analysis (if unset, then it is the last date of the downloaded period); write in format "YYYY-MM-DD" or define as None.
last_downloaded_data_date = '2022-11-30'

# Frequency of the data to download
downloaded_data_freq = '1d' # e.g. '5m' (5 minutes), '1d' (1 day).


# For nct, genhyperbolic or levy_stable distributions: If True, then skewed distributions are sought; if False, density functions are symmetric.
consider_skewness = True

# For genhyperbolic distributions, set to True if the p parameter is allowed to have nonzero values
consider_p = True

# Number of times that each set of starting parameters in the fitting of the random variable is tried. Each trial corresponds to a different first iteration of the gradient descent algorithm.
n_random_trials = 1 # Suggested value for levy_stable: 2

# Max number of iterations in each search for optimal parameters (using the gradient descent algorithm, with a given starting value for the distribution parameters).
max_n_iter = 11 # Suggested value for levy_stable: 1



# Lower limit to the correlation to discard a pair in an analysis for spread
correlation_threshold = 0.5




# ====================================================================================================
#            BLOCK FOR EVALUATION OF OPTIMAL STRATEGY
# ====================================================================================================


# Kind of quantity which is calculated. Possibilities: "heat-map" or "optimal_solution"
output_type = "heat-map"  

# Method for calculating the optimal solution (if <<output_type="optimal solution">>). Either "Barzilai-Borwein" or "bisection"
#method_optimal_solution = "Barzilai-Borwein" #"Barzilai-Borwein", "bisection" 

# Path to the file which contains the parameters of the Ornstein-Uhlenbeck equation and its residuals. Set to 'default' to use the data calculated in previous steps.
path_rv_params = 'default'#'Input_parameters/spr_fitting_params_oil_companies_norm_NCT_genhyperbolic_levy_stable.csv'

# Probability of a Poisson event. Set to None for no Poisson events.
#poisson_probability = 0.001

# Value acquired after a Poisson event. If evolution_type is "single_product" this is the value of the price itself after the event. If calculation_mode is "Ornstein-Uhlenbeck" then it is the new long-term mean (E0).    
#new_value_after_poisson_event = 0.8

# Number of periods (discrete times considered in the Ornstein-Uhlenbeck equation) to exit the position (investment) if no threshold was reached before
list_max_horizon = [9999]# [ 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91 ] 


# Lists of analysed enter value, profit-taking and stop-loss thresholds to analyse when plotting the heat-maps (2D plots to present the performance of the trading strategy)
list_enter_value   = array( list( arange(-0.02500000001, -0.01000000000, 0.001) ) +  list( arange(0.010000000, 0.025000002, 0.001) ) ) #arange(-0.08,0.08,0.01)   #arange(-0.8,-0.7,0.2) #array([-2]) # array([-3,-2,-1.8,-1.6,-1.4,-1.2,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,-0.0000001]) #array([-0.8]) #arange(-2.001,0.05,0.1) #array([-0.8]) # array([-6.0,-4.0,-2.0,0]) array([-4.0,-2.0,-1.0,-0.75,-0.5,0])
list_profit_taking = arange(0.000000001, 0.020000002, 0.001) #arange(0,0.1,0.005) # arange(0,0.3,0.025) #arange(1,21,1) arange(1,20,0.5)
list_stop_loss     = array([-9999]) #arange(-10,-9,5)  #arange(-20,-15,2) #arange(-20,0,1)  arange(-20,-1,0.5)


# Number of days that the strategy is expected to be used; in practice this means that its return is evaluated in one period with this duration (in each iteration of the Monte Carlo algorithm).
strategy_duration = 252

# Number of Monte Carlo path generated in the evaluation of each set of trading rules. This quantity is central for the numerical complexity of the calculations and for their accuracy. Set it carefully.
min_num_trials = 10000 

# Logarithm of the discount factor per unit time (e.g. per day if each row of the inputted time series corresponds to one day). If absent or set to None or to zero then discount factor is 1 (i.e. no discount factor is applied). For example, if set to log(1/(1+0.04))/365 being the periodicity one day, it corresponds to a one-year discount rate of 4%. 
log_discount_factor_unit_time =  log(1/(1+0.04))/365
discount_factor_update_method = "weekly"


# Method for the calculation of the profit. Must be in ["enter_ensured", "enter_random_one_deal", "enter_random_many_deals"].
# "enter_ensured" is the method presented in Chapter 13 of Marcos Lopez de Prado's "Advances in Financial machine learning". In it, the position is 100% sure built in t=0, and after reaching a threshold the process starts again.
# "enter_random_one_deal" means that the initial value of the spread is its average (for Ornstein-Uhlenbeck), then it randomly varies until the prescribed enter level is reached; after reaching a threshold the process starts again.
# "enter_random_many_deals" means that the initial value of the spread is its average (for Ornstein-Uhlenbeck), then it randomly varies until the prescribed enter level is reached; after this, the spread continues to evolve, and we can build new positions, whose return is added to make a cumulative return.
method_for_calculation_of_profits = "enter_random_many_deals"





# ====================================================================================================
#            BLOCK FOR OUT-OF-SAMPLE CALCULATIONS
# ====================================================================================================

# The variables below determine the parameters for a calculation of the realised profit (measured in currency units, not in spread units)

# The variable below defines how the Spread is defined. If it is "y_vs_x" then the Spread is log(Price_{label_y}) - gamma·log(Price_{label_x}), where gamma is the slope of the Return_{label_y}-vs-Return_{label_x}. If it is "x_vs_y", then the prices and returns are swapped.
oos_spread_type = "x_vs_y" # "x_vs_y" or "x_vs_y

# Labels of the products whose prices will take part in the out-of-sample calculation
oos_product_label_x = "V"
oos_product_label_y = "MA"

# Name of the list of products; it must be the one specified in <<list_product_labels>> when the download was performed.
list_product_labels_name = "credit_cards"

# First date for the out-of-sample calculation
first_oos_date =  '2019-11-30' # crypto: '2022-05-01' 

# Last date for the out-of-sample calculation
last_oos_date  =   '2022-11-29' # crypto: '2022-12-20'  # 

# Time interval prior to the out-of-sample period which forms the in-sample period (i.e. whose information is used to calibrate the random variables). Set e.g. to '100d' for 100 trading days or '2y' for two years. 
in_sample_t_interval = '5y'

# Frequency for the recalibration of the parameters of the input variable (i.e. Ornstein-Uhlenbeck parameters and parameters of the random variable corresponding to its residuals).
in_sample_recalibration_freq = '3m'           

# Choose between "Sharpe_ratio", "average", "standard_deviation", "semideviation", "Sharpe_ratio_with_semideviation", "VaR", "ES", or "probab_loss"
quantity_to_analyse = "Sharpe_ratio" #"Sharpe_ratio_with_semideviation"



###################################################################################################


# Period of the data to download
#downloaded_data_period = '12y' # e.g.  '2mo', '12y'.

# List of mean reversion parameters to analyse 
#list_E0 = [0]# [ 0, 5, 10, -5 ]

# List of half-life parameters to analyse (this determines the speed parameter phi)
#list_tau = [25]# [ 5, 10, 25, 5, 100]

# List of averages of residuals to analyse
#list_mu = [ 0 ] 

# List of standard deviations of the residuals to analyse
#list_sigma = [ 1 ]

