'''
This is the main script for the calculations on a possible paper to apply the profit-taking and stop-loss
strategies presented in chapter 13 of "Advances in financial ML", by Marcos Lopez de Prado, together with the 
use of fat-tailed distributions, as done in Göncü & Erdinc Akyildirim "A stochastic model for commodity pairs trading".
This code is expected to provide the optimal levels for stop-loss and profit-taking thresholds for products governed
either by: i) A variation of the Ornstein-Uhlenbeck (OU) equation where the random variable is fat-tailed (this
is customarily applied to pairs trading, being the main variable a PRICE), and ii) To products whose RETURNS have a given
distribution.
The user must provide the following inputs:
In an input_parameters.py file:
*
*

In a .csv file, the time series of the prices of the product.

Part of the code below is a modified version of the code which appears on page 175 of "Advances in Financial Machine
Learning" (Marcos Lopez de Prado), downloadable from https://quantresearch.org/OTR.py.txt

References
[1] Marcos Lopez de Prado, "Advances in Financial Machine Learning", Chap. 13, Wiley, 2018.
[2] Ahmet Göncü & Erdinc Akyildirim, "A stochastic model for commodity pairs trading", Quantitative Finance, 2016.

'''


# ====================================================================================================
#       BLOCK FOR DOWNLOADING DATA AND FITTING THEM TO PROBABILITY DISTRIBUTIONS
# ====================================================================================================


#Initialization

from module_initialization import InputParams

input_params = InputParams()



#from module_plots import plot_histograms_without_fitting; plot_histograms_without_fitting(input_params); exit(0)

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




'''TRASH / OLD CODE:


from module_plots import plot_2D 
import numpy as np
output_dir = '/Users/pgr/Desktop/Finanzas/Papers_Finanzas_IM/Paper2_Stopping_strategy/tex/_latest_version/Figures/Poisson_pairs/'
for distrib in ['nct']:
    datafile_path = output_dir+'OrnUhl-spr_resid_BP_TTE_y_vs_x-'+distrib+'-wiPoisson-50k__sl_-0p15.csv'
    plot_2D( datafile_path, output_dir, "BP_TTE_Spread_y_vs_x", "profit_taking_param", "Sharpe_ratio", "Sharpe_ratio_with_semideviation", 1/np.sqrt(2), "BP / TTE ;  t-student distrib." )
exit(0)


'''

 