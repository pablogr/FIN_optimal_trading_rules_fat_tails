
'''
This module contains parameters which are used by the functions in the calculations of the optimal stopping of a trading
strategy. Note that the parameters here are just a part of the parameters used in a simulation, the rest being
stored in the input.py file which lies in the directory where the calculations are run.
'''

tolerance_for_heatmap = 0.000001  # Tolerance to stop the calculation corresponding to a single pair of (profit-taking,stop-loss) thresholds in a calculation which plots a heatmap.
tolerance_for_optimum = 0.001     # Tolerance to stop the calculation corresponding to a single pair of (profit-taking,stop-loss) thresholds in a calculation which seeks the optimum. Note that convergence in the Sharpe ratio below 0.1% could be impossible to attain.
max_iter_search_optimum = 20      # Maximum number of iterations in the search for the optimum profit-taking and stop-loss thresholds
min_iter_search_optimum = 2       # Minimum number of iterations in the search for the optimum profit-taking and stop-loss thresholds
tolerance_search_optimum = 0.0005  # This is the difference between the threshold (profit-taking, stop-loss) in consecutive iterations of the search for the optimum which makes the iterative solver stop.
alpha_for_VaR_and_ES  = 0.01      # This is the alpha to be used in the calculation of the Value-at-Risk and Expected Shortfall.

dict_text_opt_method = {'Barzilai-Borwein': 'gradient descent (Barzilai-Borwein)','bisection': 'bisection (WARNING: This is much slower than "Barzilai-Borwein")'}

plot_labels = {"max_horizon": "maximum horizon (days)","enter_value": "Enter value ($)", "profit_taking_param": "profit-taking ($)",
            "stop_loss_param": "stop-loss ($)", "profit_mean": "average profit",
            "profit_std": "standard deviation of profit", "Sharpe_ratio": "$SR$",
            "semideviation": "Semi-deviation", "Sharpe_ratio_with_semideviation": "$SR'$",
            "VaR": "VaR", "ES": "Expected Shortfall", "probab_loss": "Probability of loss"}

list_columns = ["profit_mean", "profit_std", "Sharpe_ratio", "Sharpe_ratio_with_semideviation", "semideviation", "VaR","ES", "probab_loss"]

dict_price_types = {"Open":"opening",  "Close":"closing", "High":"maximum", "Low":"minimum" }

distribution_types = frozenset( ('norm', 'nct', 'genhyperbolic', 'levy_stable') )
distribution_types_text = { 'norm':'normal', 'nct':'non-centered t-student', 'genhyperbolic':'generalized hyperbolic','levy_stable':'levy stable' }
distribution_types_text2 = { 'norm':'normal', 'nct':'nct', 'genhyperbolic':'ghyp','levy_stable':'stable' }
distribution_types_text3 = { 'norm':'Normal', 'nct':'t-student', 'genhyperbolic':'Gen. hyperbolic','levy_stable':'Stable' }


tolerance_fitting = 10**(-7)

