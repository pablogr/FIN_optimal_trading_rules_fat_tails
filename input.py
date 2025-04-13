# This is the file which contains the parameters defined by the user for the analysis of optimal trading strategies through the
# seek for enter value, profit-taking and stop-loss thresholds. See the article "The effect of fat tails on rules for optimal pairs
# trading: profit implication of regime switching with Poisson events" (P. Garcia-Risueno et al, 2025).


from os import getcwd
from numpy import array, arange


# Directory where the file with user-defined input parameters input.py is stored
input_directory  = getcwd()


# ====================================================================================================
#              GENERAL INPUT PARAMETERS (for both blocks)
# ====================================================================================================

# Kind of calculations to be performed. Set to either "download", "fit", "download_and_fit", "find_trading_rules" or "out_of_sample". "download" downloads the data from Yahoo Finance; "fit" fits the data to probability distributions (and eventually to the Ornstein-Uhlenbeck equation), "download_and_fit" makes both; "find_trading_rules" calculates the optimal trading rules; "out_of_sample" calculates the profitability in specified periods.
calculation_mode =  "download_and_fit"

regime_switching = False # If True, regime switching a la Hamilton 1989 is applied to the mean reversion parameter E0. This makes the calculations much slower than if regime_switching=False, hence set it to True only if looking at the plot of the spread it is clear that there is a regime switching, which can be identified as a series having two clearly different levels. Note that if you set it to True, just norm and nct distributions for the residuals are possible. This affects both the calculations of searching the parameters of the distribution (E0, sigma, ...) AND the calculations of finding optimal trading rules.
#spread_type_for_regime_switching = "y_vs_x" # "x_vs_y" or "y_vs_x" # This variable sets how the Spread is defined. If it is "y_vs_x" then the Spread is log(Price_{label_y}) - gamma·log(Price_{label_x}), where gamma is the slope of the Return_{label_y}-vs-Return_{label_x}. If it is "x_vs_y", then the prices and returns are swapped. Since fitting is slow in regime switching calculations, if regime_switching is set to True, then only y_vs_x or x_vs_y is analysed.

# The variable below indicates the type of trading that will be performed. Possibilities: "Ornstein-Uhlenbeck_equation" (for pairs trading) or "single_product" (for single-product trading).
evolution_type = "Ornstein-Uhlenbeck_equation"

# Types of the probability distributions to fit the random variables. It must be either 'norm', 'nct', 'johnsonsu', 'genhyperbolic' or 'levy_stable'.
list_distribution_types = [ 'norm', 'nct']

# Indices to download. They will be used in the calculations of the stationary parts.
list_index_labels = ['^STOXX', 'BZ=F']   #
# '^GSPC': SP500
# '^STOXX': EuroStoxx600
# '^DJI': Dow Jones Index
# 'BZ=F''^GSPC': Brent oil barrel
# 'BITW':  Bitwise 10 Crypto Index Fund
# '^BCOM': Bloomberg commodities index


# Labels of the products whose time-series must be downloaded. Note that they are taken into account only if "download" is in calculation_mode; otherwise the starting point of the calculations is the stationarity file (in the Output directory).
# Each list_product_labels consists of 4 parts:
#  i)   The name of the set of products (e.g. US_banks; this is an arbitrary label);
#  ii)  The list of Yahoo Finance tickers of the stocks of cryptocurrencies whose pairs will be analysed;
#  iii) The list of the tickers risk factors (e.g. '^STOXX50E' for the EuroStoxx50) to be used in the calculation of
#       common trends to calculate the gamma to be used in the definition of the spread.
#  iv)  The name of the file which contains the risk-free_rate-vs-t data (located in the Time_series directory,
#       e.g. 'US_1Y_yields' or 'GER_1Y_yields', which can be manually downloaded from https://www.investing.com/rates-bonds/germany-1-year-bond-yield-historical-data and https://www.investing.com/rates-bonds/u.s.-1-year-bond-yield-historical-data).
#       THE RISK-FREE RATE MUST BE EXPRESSED IN % (e.g. 5.1 for 5.1%, not 0.051). This is optional, if omitted the risk-free rate
#       will be assumed to be 0 for every time, this is it will not be used in the calculation of the gamma which defines the spread.
#

list_product_labels =  [ "oil_companies", [ "PSX", "TTE" ], ['^STOXX', 'BZ=F'], 'GER_1Y_yields'  ]
#list_product_labels = ["discretionary_US",[ "BKNG","MAR" ],['^GSPC',], 'US_1Y_yields']
#list_product_labels = [ "materials", [ "BNR.DE","UPM.HE"],  [ '^STOXX','^BCOM'], 'GER_1Y_yields' ]
#list_product_labels =  [ "oil_companies", ["BP", "TTE" ], ['^STOXX', 'BZ=F'], 'GER_1Y_yields'  ]
#list_product_labels = ["banks_USA",[ "BAC", "PNC" ],['^GSPC'], 'US_1Y_yields']
#list_product_labels = [ "health", [ "AMGN", "SYK" ],[ '^GSPC'], 'US_1Y_yields' ]
#list_product_labels = ["utilities_US",[ "DUK", "SRE"],['^GSPC' ], 'US_1Y_yields']
#list_product_labels = [ "software",[  "GOOG","INTU" ],[ '^GSPC' ], 'US_1Y_yields']
#list_product_labels = ["real_estate_USA",["EQR", "ESS" ],['^GSPC'], 'US_1Y_yields']
#list_product_labels = ["communications_US",[  "DIS","VZ" ],['^GSPC'], 'US_1Y_yields']
#list_product_labels = ["staples_US", [ "MDLZ","MNST" ],['^GSPC'], 'US_1Y_yields']
#list_product_labels = ["industrials_US",[  "CAT","RELX"  ],['^GSPC','^DJI'], 'US_1Y_yields']
#list_product_labels = [ "cryptocurrencies", ["BTC-USD","TRX-USD"],[ 'BITW'], 'US_1Y_yields']
#list_product_labels = [ "cryptocurrencies2", ["XRP-USD", "DOGE-USD" ],['BITW'], 'US_1Y_yields']


# LIST OF SETS WITH STATIONARY PAIRS:
#FOR REGIME SWITCHING:list_product_labels =  [ "oil_companies", ["PSX","TTE" ], ['^STOXX', 'BZ=F'], 'GER_1Y_yields'  ] # '^STOXX50E' [ "XOM", "CVX", "COP", "PSX", "VLO", "MPC", "MRO", "EOG", "BP", "SHEL", "TTE", "E", "EQNR", "CNQ", "OXY", "NTGY.MC" ], 'GER_1Y_yields' ] #, "BZ=F"  # Oil companies (worldwide) sold in NY; MPC and PSX are available just from 2011 and 2012, respectively.  "BZ=F" is the oil barrel today, I think.
#list_product_labels =  [ "oil_companies", ["SHEL", "BP"], ['^STOXX', 'BZ=F'], 'GER_1Y_yields'  ]
#list_product_labels =  [ "oil_companies", [  "CVX", "TTE" ], ['^STOXX', 'BZ=F'], 'GER_1Y_yields'  ] # '^STOXX50E' [ "XOM", "CVX", "COP", "PSX", "VLO", "MPC", "MRO", "EOG", "BP", "SHEL", "TTE", "E", "EQNR", "CNQ", "OXY", "NTGY.MC" ], 'GER_1Y_yields' ] #, "BZ=F"  # Oil companies (worldwide) sold in NY; MPC and PSX are available just from 2011 and 2012, respectively.  "BZ=F" is the oil barrel today, I think.
#list_product_labels =  [ "oil_companies", [  "CVX", "TTE" ], ['^GSPC','CL=F'], 'US_1Y_yields'  ]
#list_product_labels =  [ "oil_companies", ["SHEL", "BP", "XOM", "CVX", "COP", "PSX", "VLO", "MPC",  "EOG", "TTE", "E", "EQNR", "CNQ", "OXY", "NTGY.MC" ], ['^STOXX', 'BZ=F'], 'GER_1Y_yields'  ] # '^STOXX50E' [ "XOM", "CVX", "COP", "PSX", "VLO", "MPC", "MRO", "EOG", "BP", "SHEL", "TTE", "E", "EQNR", "CNQ", "OXY", "NTGY.MC" ], 'GER_1Y_yields' ] #, "BZ=F"  # Oil companies (worldwide) sold in NY; MPC and PSX are available just from 2011 and 2012, respectively.  "BZ=F" is the oil barrel today, I think.
#list_product_labels = ["banks_USA",["BAC","PNC"],['^GSPC'], 'US_1Y_yields']
#list_product_labels = ["banks_USA",["GS","C","BAC","MS","JPM","BLK","BX","WFC","USB","TFC","PNC","COF","BK"],['^GSPC'], 'US_1Y_yields']
#list_product_labels = ["banks_USA",[ "BAC", "PNC" ],['^GSPC'], 'US_1Y_yields']
#list_product_labels = ["computer_hardware",["ASML.AS","AMD","NVDA","DELL","INTC","TXN","IBM"],['^IXIC'], 'US_1Y_yields']
#list_product_labels = ["retailers_USA",["WMT","WBA","COST","HD","KR","BBY","TGT"],['^GSPC','^DJI'], 'US_1Y_yields']
#list_product_labels = [ "electric_power_companies", ["ENGI.PA", "D2G.DE", "ENEL.MI", "IBE.MC", "ELE.MC", "RED.MC", "NTGY.MC", "EOAN.DE", "RWE.DE", "EBK.DE", "SSE.L", "EDP.LS" ], ['^STOXX'], 'GER_1Y_yields' ]#[ "electric_power_companies", ["ENGI.PA", "D2G.DE", "ENEL.MI", "IBE.MC", "ELE.MC", "RED.MC", "NTGY.MC", "EOAN.DE", "RWE.DE", "EBK.DE", "SSE.L", "EDP.LS" ] ]# European electricity companies. Note that SSE is in pounds!
#list_product_labels = ["nasdaq_composite",["T","TMUS","TXN","IBM","AVGO","CSCO","ADBE"],['^IXIC'], 'US_1Y_yields']
#list_product_labels = [ "miners", ["RIO","AAL.L","BHP","GLEN.L", "FXC", "FMG.AX","VALE"],  [ '^STOXX','^GSPC','^BCOM'], 'GER_1Y_yields' ] # v1: ['^FTSE' , '^BCOM','^STOXX'], 'GER_1Y_yields'  ]  ;  'TIO=F','GD=F','^BCOM'
#list_product_labels = [ "gold_miners", ["GOLD","NEM","AU","KGC","GFI","AEM","PLZL.ME","RIO","AAL.L","BHP","GLEN.L"],  [ '^BCOM','GC=F'], 'GER_1Y_yields' ] # v1: ['^FTSE' , '^BCOM','^STOXX'], 'GER_1Y_yields'  ]  ;  'TIO=F','GD=F','^BCOM'
#list_product_labels = [ "cryptocurrencies", ["BTC-USD","BNB-USD","ETH-USD","SOL-USD","ADA-USD","XRP-USD","DOT-USD","DOGE-USD","XMR-USD","AVAX-USD","TRX-USD"],['^CMC200','^IXIC'], 'US_1Y_yields'] #'^GSPC''^IXIC'
#list_product_labels = ["retailers_USA2",[ "WMT","COST","PG","KO","PEP","UL","BUD","MO","MDLZ","CL","DEO","TGT" ],['^GSPC','^DJI'], 'US_1Y_yields']
#list_product_labels = ["industrials_US",[ "GE","RTX","CAT","UNP","HON","BA","DE","ETN","LMT","UPS","RELX" ,"WM",""PH,"CTAS" ,"TT","MMM","TRI" ,"ITW","TDG","CP","TDG","RSG","GD","EMR" ],['^GSPC','^DJI'], 'US_1Y_yields']
#list_product_labels = ["materials_US",[ "LIN","SHW","SCCO","ECL","APD","CRH","FCX","NEM","AEM","CTVA","VMC" ],['^GSPC','^DJI'], 'US_1Y_yields']
#list_product_labels = ["communications_US",[ "NFLX","TMUS","DIS","T","VZ","CMCSA","SPOT","GOOGL","AMX", "LIV" ],['^GSPC'], 'US_1Y_yields']
#list_product_labels = ["communications_EU",[ "DTE.DE","FTE.DE" ,"TEF.MC","BT-A.L","TIT.MI"],['^STOXX'], 'GER_1Y_yields']
#list_product_labels = ["real_state_USA2",[ "PLD","WELL","EQIX","AMT","SPG","DLR","PSA","O","CBRE","CCI","EXR" ,"VICI","CSGP","AVB","VTR","EQR","IRM" ],['^GSPC'], 'US_1Y_yields']




# Set to more than 0 (i.e. 0.5, 1 or 2) to print partial results from screen
verbose = 0.5

make_plots = True
only_plots = False


# ====================================================================================================
#       BLOCK FOR DOWNLOADING DATA AND FITTING THEM TO PROBABILITY DISTRIBUTIONS
# ====================================================================================================

#This is the file which contains the parameters defined by the user for the
# downloading financial time series and fitting them to appropriate distributions.

# First date for our analysis (if unset, then it is the first date of the downloaded period); write in format "YYYY-MM-DD" or define as None.
first_downloaded_data_date =  '2017-02-20'

# Last date for our analysis (if unset, then it is the last date of the downloaded period); write in format "YYYY-MM-DD" or define as None.
last_downloaded_data_date =   '2025-02-20'


# Frequency of the data to download
downloaded_data_freq = '1d' # e.g. '5m' (5 minutes), '1d' (1 day).


#For nct, genhyperbolic or levy_stable distributions: If True, then skewed distributions are sought; if False, density functions are symmetric.
consider_skewness = True

# For genhyperbolic distributions, set to True if the p parameter is allowed to have nonzero values
consider_p = True

# Number of times that each set of starting parameters in the fitting of the random variable is tried. Each trial corresponds to a different first iteration of the gradient descent algorithm.
n_random_trials = 2 # Suggested value for levy_stable: 2

# Max number of iterations in each search for optimal parameters (using the gradient descent algorithm, with a given starting value for the distribution parameters).
max_n_iter = 20 #  Suggested value for levy_stable: 1

# Lower limit to the correlation to discard a pair in an analysis for spread
correlation_threshold = 0.25





# ====================================================================================================
#            BLOCK FOR EVALUATION OF OPTIMAL STRATEGY
# ====================================================================================================

#Kind of quantity which is calculated. Possibilities: "heat-map" or "optimal_solution" #
output_type = "heat-map"

# Method for calculating the optimal solution (if <<output_type="optimal solution">>). Either "Barzilai-Borwein" or "bisection"
#method_optimal_solution = "Barzilai-Borwein" #"Barzilai-Borwein", "bisection"

# Path to the file which contains the parameters of the Ornstein-Uhlenbeck equation and its residuals. Set to 'default' to use the data calculated in previous steps.
path_rv_params = 'default' #Alternative: 'Input_parameters/spr_fitting_params_oil_companies_norm_NCT_genhyperbolic_levy_stable.csv'

# ===== BLOCK FOR INPUT PARAMETERS WITH NON-CONSTANT MEAN (E0, VALUE THE TIME SERIES REVERTS TO) USING POISSON EVENTS =======================
#
# Change of the mean modelled through Poisson changes (jumps of E0).
# and does require to manually introduce the two values of E0 or the jump probabilities. Note that this Poisson changes
# only affect the calculations for finding optimal trading rules, NOT the fitting of parameters.

# Probability of a Poisson event. Set undefined for no Poisson events.
#poisson_probability =  None# 0.0004329 #old bien para Poisson: 0.01

# Value acquired after a Poisson event. If evolution_type is "single_product" this is the value of the price itself after the event. If calculation_mode is "Ornstein-Uhlenbeck" then it is the new long-term mean (E0).
#new_value_after_poisson_event_increase = None # 0.62  # This makes E0 to increase if there is a Poisson event (with probability 1/2).
#new_value_after_poisson_event_decrease = None # -0.62 # This makes E0 to decrease if there is a Poisson event (with probability 1/2).

# === End of the block of input parameters with non-constant mean a la Poisson ======================================================


# Number of periods (discrete times considered in the Ornstein-Uhlenbeck equation) to exit the position (investment) if no threshold was reached before
list_max_horizon = [252*2]   


# Lists of analysed enter value, profit-taking and stop-loss thresholds to analyse when plotting the heat-maps (2D plots to present the performance of the trading strategy). Note that the numberes below are additive to the mean reversion parameter E0 (or, if there is regime switching a la Hamilton, additive to the lowest mean reversion parameter alpha0).

if( calculation_mode == "out_of_sample"):
    list_enter_value   = array( list( arange(-0.2000001, -0.0000000001, 0.01) ) +  list( arange(0.000000001, 0.200002, 0.01) ) )
    list_profit_taking = list_profit_taking = arange( -0.2, 0.0, 0.01)
    list_stop_loss     = array([0.4])
else:
    list_enter_value   =  arange(0.0000000001, 0.080001, 0.004)   
    list_profit_taking =  arange( -0.1,  0.000001, 0.005)
    list_stop_loss     =  array([999])


# Number of days that the strategy is expected to be used; in practice this means that its return is evaluated in one period with this duration (in each iteration of the Monte Carlo algorithm).
strategy_duration = 252*2

# Number of Monte Carlo path generated in the evaluation of each set of trading rules. This quantity is central for the numerical complexity of the calculations and for their accuracy. Set it carefully.
min_num_trials = 200000


# Method for the calculation of the profit. Must be in ["enter_ensured", "enter_random_one_deal", "enter_random_many_deals"].
# "enter_ensured" is the method presented in Chapter 13 of Marcos Lopez de Prado's "Advances in Financial machine learning". In it, the position is 100% sure built in t=0, and after reaching a threshold the process starts again.
# "enter_random_one_deal" means that the initial value of the spread is its average (for Ornstein-Uhlenbeck), then it randomly varies until the prescribed enter level is reached; after reaching a threshold the process starts again.
# "enter_random_many_deals" means that the initial value of the spread is its average (for Ornstein-Uhlenbeck), then it randomly varies until the prescribed enter level is reached; after this, the spread continues to evolve, and we can build new positions, whose return is added to make a cumulative return.
method_for_calculation_of_profits = "enter_random_many_deals"





# ====================================================================================================
#            BLOCK FOR OUT-OF-SAMPLE CALCULATIONS
# ====================================================================================================

# The variables below determine the parameters for a calculation of the realised profit (measured in currency units, not in spread units)

# The variable below must be either "x_vs_y" or "x_vs_y" . It defines how the Spread is defined. If it is "y_vs_x" then the Spread is log(Price_{label_y}) - gamma·log(Price_{label_x}), where gamma is the slope of the Return_{label_y}-vs-Return_{label_x}. If it is "x_vs_y", then the prices and returns are swapped.
oos_spread_type = "y_vs_x"

# WARNING: Make sure that the product labels and list of products labels coincide with some of the values defined above (in the download block of input.py).
# Labels of the products whose prices will take part in the out-of-sample calculation
oos_product_label_x =  "TTE"
oos_product_label_y =  "PSX"

# Name of the list of products; it must be the one specified in <<list_product_labels>> when the download was performed.
list_product_labels_name = "oil_companies"

# First date for the out-of-sample calculation
first_oos_date =  '2021-02-20'  

# Last date for the out-of-sample calculation
last_oos_date  = '2025-02-19'

# Time interval prior to the out-of-sample period which forms the in-sample period (i.e. whose information is used to calibrate the random variables). Set e.g. to '100d' for 100 trading days or '2y' for two years.
in_sample_t_interval = '5y'  

# Frequency for the recalibration of the parameters of the input variable (i.e. Ornstein-Uhlenbeck parameters and parameters of the random variable corresponding to its residuals).
in_sample_recalibration_freq =   '3m'

# Choose between "Sharpe_ratio", "average", "standard_deviation", "semideviation", "Sharpe_ratio_with_semideviation", "VaR", "ES", or "probab_loss"
quantity_to_analyse = "Sharpe_ratio" #"Sharpe_ratio_with_semideviation"

# Rate for (continuously compounding) discount factors. Set e.g. to 0.01 for 1%. If you do not want to consider discount factors, set it to None.
discount_rate = None

# Rate for transaction costs (in dollars, NOT in spread; see explanations in the paper). Set e.g. to 0.01 for 1%. If you do not want to consider transaction costs, set it to None.
transaction_costs_rate = 0.005



