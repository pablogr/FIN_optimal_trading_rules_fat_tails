'''
This is the main script for the calculations presented in the article "The effect of fat tails on rules for optimal pairs
trading: profit implication of regime switching with Poisson events" (P. Garcia-Risueno et al, 2025).
This code provides the optimal levels for enter value, stop-loss and profit-taking thresholds for products governed
either by: i) A variation of the Ornstein-Uhlenbeck (OU) equation where the random variable is fat-tailed (this
is customarily applied to pairs trading, being the main variable a PRICE), and ii) To products whose RETURNS have a given
distribution.
The user must provide the following inputs:
* In an input.py file:
* If a risk-free rate will be considered (e.g. 'US_1Y_yields', see input.py), then the file containing its data.
The time series of the analysed financial products and benchmarks will be automatically downloaded, as long as there is an internet connection.

The seed of the code below appears on page 175 of "Advances in Financial Machine
Learning" (Marcos Lopez de Prado), downloadable from https://quantresearch.org/OTR.py.txt

Important references:
[1] Marcos Lopez de Prado, "Advances in Financial Machine Learning", Chap. 13, Wiley, 2018.
[2] Ahmet Göncü & Erdinc Akyildirim, "A stochastic model for commodity pairs trading", Quantitative Finance, 2016.

'''

# ====================================================================================================
#       BLOCK FOR DOWNLOADING DATA AND FITTING THEM TO PROBABILITY DISTRIBUTIONS
# ====================================================================================================


#Initialization
import numpy as np
np.random.seed(seed=1234)

from module_initialization import InputParams

input_params = InputParams()


# ====================================================================================================
#           BLOCK FOR DOWNLOADING TIME-SERIES AND FITTING THEM TO PROBABILITY DISTRIBUTIONS
# ====================================================================================================

# Download data and calculation of correlations, stationarity and spreads of pairs (if required)

if (input_params.download_data):

    from module_download_ts import download_ts
    from module_fitting import find_pais_hi_correlation, calc_all_spreads

    if (input_params.download_data):
        download_ts( input_params )
        find_pais_hi_correlation(input_params)
        calc_all_spreads(input_params)


# Fitting data to probability distribution and making the corresponding plots

if (input_params.fit_data):

    from module_fitting import fit_residuals
    from module_plots import make_all_plots_2D

    fit_residuals( input_params )
    make_all_plots_2D( input_params)



# ====================================================================================================
#             BLOCK FOR EVALUATION OF OPTIMAL TRADING RULES
# ====================================================================================================

if ( input_params.calculation_mode in ["find_trading_rules","download_fit_and_find_trading_rules"]):

    from module_sweep import sweep_enptsl_parameters
    
    sweep_enptsl_parameters( input_params )




# ====================================================================================================
#           BLOCK FOR CALCULATION OF OUT-OF-SAMPLE PROFITS
# ====================================================================================================

from module_out_of_sample_calc import calculate_oos_profits


if ( input_params.calculation_mode == "out_of_sample"):
    calculate_oos_profits( input_params )


#----

print("\n **** The calculations finished satisfactorily ****\n")

