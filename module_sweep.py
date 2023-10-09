'''
This module contains the functions for sweeping the parameters in the calculations which determine the optimal stopping
of a trading strategy, as well as functions called by them.
'''


import pandas as pd
from module_find_data_heatmap import find_thresholds_heatmap
from module_find_best_strategy import find_optimal_thresholds
from module_plots import save_and_plot_results, plot_heatmaps_pt_vs_en
from numpy import log2
from module_parameters import distribution_types_text

#----------------------------------------------------------------------------------------

def read_rv_params(input_params, prod_label, distr ):
    '''This function reads the parameters of the random variable (and eventually also of the Ornstein-Uhlenbeck
    equation) from the corresponding file which stores them.'''

    df_prod = input_params.df_rv_params.loc[prod_label]
    coddistr = {"Gaussian":'norm', "norm":'norm', "normal":'norm',"nct":'nct', "tstudent":'nct', 'genhyperbolic':'genhyperbolic', 'stable':'levy_stable', 'levy_stable':'levy_stable' }
    distr = coddistr[distr]

    if (input_params.evolution_type == "Ornstein-Uhlenbeck_equation"):
        OU_params = {'product_label':prod_label, 'E0': df_prod['OU_E0'], 'phi': df_prod['OU_phi'], 'tau': -1 / (log2(df_prod['OU_phi']))}
        print(" * For "+prod_label+" the Ornstein-Uhlenbeck equation has the parameters that follow:\n   E0=",OU_params['E0'],"; phi=",OU_params['phi'],"; tau=",OU_params['tau'])
        text = "spread residuals"
    else:
        OU_params = None
        text = "returns"

    if (distr == 'norm'):
        rv_params = {'product_label':prod_label, 'distribution_type': distr, 'mu': df_prod["normal_loc"], 'sigma': df_prod["normal_scale"], 'third_param': None,'fourth_param': None, 'fifth_param': None}
        print("   The Gaussian distribution of "+text+" has: mu=","{:.10f}".format(rv_params['mu']),"; sigma=","{:.8f}".format(rv_params['sigma']),"\n")
    elif (distr =='nct'):
        rv_params = {'product_label':prod_label, 'distribution_type': distr,'mu': df_prod["nct_loc"], 'sigma': df_prod["nct_scale"], 'third_param': df_prod["nct_skparam"],'fourth_param': df_prod["nct_dfparam"], 'fifth_param': None}
        print("   The non-centered t-student distribution of " + text + " has: mu=", "{:.10f}".format(rv_params['mu']), "; sigma=", "{:.8f}".format(rv_params['sigma']), "; skewness param.=","{:.6f}".format(rv_params['third_param']),"; num. DoF param=","{:.4f}".format(rv_params['fourth_param']),"\n")
    elif (distr == 'genhyperbolic'):
        rv_params = {'product_label':prod_label, 'distribution_type': distr,'mu': df_prod["ghyp_loc"], 'sigma': df_prod["ghyp_scale"], 'third_param': df_prod["ghyp_b_param"],'fourth_param': df_prod["ghyp_a_param"], 'fifth_param': df_prod["ghyp_p_param"]}
        print("   The generalized hyperbolic distribution of " + text + " has: mu=", "{:.10f}".format(rv_params['mu']), "; sigma=", "{:.8f}".format(rv_params['sigma']),"; b (skewness) param.=", "{:.6f}".format(rv_params['third_param']), "; a param.=", "{:.4f}".format(rv_params['fourth_param']), "; p param.=", "{:.4f}".format(rv_params['fifth_param']),"\n")
    elif (distr == 'levy_stable'):
        rv_params = {'product_label':prod_label, 'distribution_type': distr,'mu': df_prod["stable_loc"], 'sigma': df_prod["stable_scale"], 'third_param': df_prod["stable_beta_param"], 'fourth_param': df_prod["stable_alpha_param"],'fifth_param': None}
        print("   The stable distribution of " + text + " has: mu=", "{:.10f}".format(rv_params['mu']), "; sigma=","{:.8f}".format(rv_params['sigma']), "; beta (skewness) param.=", "{:.6f}".format(rv_params['third_param']), "; tail param.=","{:.4f}".format(rv_params['fourth_param']),"\n")
    else:
        raise Exception("\nERROR: Unrecognized distribution type (" + distr + ").\n")

    del input_params; del prod_label; del df_prod; del distr; del coddistr

    return OU_params, rv_params


# ----------------------------------------------------------------------------------------

def sweep_enptsl_parameters( input_params ):
    ''' This function performs the calculation of finding the optimal trading strategy, i.e. to find the profit-taking
    and stop-loss thresholds, for given input parameters. input_params belongs to a class which contains the following fields (among others):

    :param list_E0: List of the values of the mean-reversion (E0) parameter of the O-U equation.
    :param list_tau: List of values of the half-life parameter (which determines the speed parameter) in the O-U equation.
    :param list_sigma: List of values of the standard deviation of the random variable of the O-U equation.
    :param list_profit_taking: List of profit-taking thresholds analysed.
    :param list_stop_loss: List of stop-loss thresholds analysed.
    :param max_horizon: Number of periods (discrete times considered in the Ornstein-Uhlenbeck equation) to exit the position (investment) if no threshold was reached before
    :param min_num_trials: Minimum number of calculations to evaluate the heat-map (set to a high value, e.g. 100000, and make sure that your results are converged in this quantity)
    :param output_directory: path to the directory where results are stored
    :return:
    '''

    df_out = pd.DataFrame()

    if (input_params.path_rv_params==None): # Parameters of the random variable read from input.py
    
        list_E0       = input_params.list_E0
        list_tau      = input_params.list_tau

        list_mu       = input_params.list_mu
        list_sigma    = input_params.list_sigma
        list_3rdparam = input_params.list_3rdparam
        list_4thparam = input_params.list_4thparam
        list_5thparam = input_params.list_5thparam

        if (input_params.evolution_type == "Ornstein-Uhlenbeck_equation"):
            for distrib in input_params.list_distribution_types:
                for E0 in list_E0:
                    for tau in list_tau:
                        #for mu, sigma, third_param, fourth_param, fifth_param in product(list_mu, list_sigma, list_3rdparam, list_4thparam, list_5thparam):
                        for mu in list_mu:
                            for sigma in list_sigma:
                                for third_param in list_3rdparam:
                                    for fourth_param in list_4thparam:
                                        for fifth_param in list_5thparam:

                                            OU_params = {'E0':E0, 'tau':tau, 'phi':2**(-1./tau) }                                # Parameters of the Ornstein-Uhlenbeck equation (see eq. (13.2) of Advances in Financial Machine Learning, by Marcos Lopez de Prado, 2018).
                                            rv_params = {'distribution_type':distrib, 'mu':mu, 'sigma': sigma, 'third_param':third_param, 'fourth_param':fourth_param, 'fifth_param':fifth_param  }   # Parameters which define the random variable to use

                                            if (input_params.output_type == "heat-map"):
                                                if not (input_params.only_plots):
                                                    df_out = find_thresholds_heatmap( input_params, rv_params, OU_params )
                                                find_thresholds_heatmap(input_params, rv_params, df_out, OU_params )
                                            elif (input_params.output_type == "optimal_solution"):
                                                find_optimal_thresholds(input_params, rv_params, OU_params)

        else: # input_params.evolution_type is "single_product"

            #for mu, sigma, third_param, fourth_param, fifth_param in product(list_mu, list_sigma, list_3rdparam,list_4thparam, list_5thparam):
            for mu in list_mu:
                for sigma in list_sigma:
                    for third_param in list_3rdparam:
                        for fourth_param in list_4thparam:
                            for fifth_param in list_5thparam:

                                for distrib in input_params.list_distribution_types:

                                        rv_params = {'distribution_type':distrib, 'mu': mu, 'sigma': sigma, 'third_param': third_param, 'fourth_param': fourth_param,'fifth_param': fifth_param}  # Parameters which define the random variable to use

                                        if (input_params.output_type == "heat-map"):
                                            if not (input_params.only_plots):
                                                df_out = find_thresholds_heatmap( input_params, rv_params  )
                                            save_and_plot_results(input_params, rv_params, df_out )
                                        elif (input_params.output_type == "optimal_solution"):
                                            find_optimal_thresholds(input_params, rv_params  )
    
    else:  # parameters of the random variable are read from file
    
        for prod_label in input_params.df_rv_params.index:

            if (input_params.check_convergence_trading_rules): (pd.DataFrame(columns=["filepath"])).to_csv(input_params.dir_trading_rules_convergence + "/list_files_to_analyse_for_convergence.csv", index=False)

            for distr_type in input_params.list_distribution_types:

                print("\n *** Now analysing",prod_label,"with probability distributions of type",distribution_types_text[distr_type]," ***\n")

                OU_params, rv_params = read_rv_params( input_params, prod_label, distr_type )
                if (input_params.output_type == "heat-map"):
                    if not (input_params.only_plots):
                        df_out = find_thresholds_heatmap( input_params, rv_params, OU_params )
                    save_and_plot_results(input_params, rv_params, df_out, OU_params )
                elif (input_params.output_type == "optimal_solution"):
                    find_optimal_thresholds(input_params, rv_params, OU_params)

            if (input_params.check_convergence_trading_rules):
                from module_plots import make_plots_convergence
                make_plots_convergence(input_params, prod_label, distr_type)

        if (input_params.make_plots): plot_heatmaps_pt_vs_en(input_params)

    del input_params; del OU_params; del rv_params
    
    return

#----------------------------------------------------------------------------------------




#----------------------------------------------------------------------------------------




