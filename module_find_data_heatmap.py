import pandas as pd
import gc
import module_parameters
from module_generic_functions import find_thresholds, adapt_lists_thresholds

#pd.set_option('max_columns', 20); pd.set_option('max_rows', 99999)

#---------------------------------------------------------------------------------------------------------------------

def create_aux_df_heatmap( list_max_horizon, en, li1, li2 ):
    '''This function creates an input dataframe with indices corresponding to the profit-taking and stop-loss values.'''

    if (en!=None):
        df0 = pd.DataFrame(columns=["enter_value","max_horizon","profit_taking_param", "stop_loss_param", 'profit_mean', 'profit_std', 'Sharpe_ratio','semideviation', 'Sharpe_ratio_with_semideviation','VaR', 'ES', 'probab_loss'])
        for l in range(len(list_max_horizon)):
            for i in range(len(li1)):
                for j in range(len(li2)):
                    my_index = l*(len(li1)*len(li2)) + i*(len(li2)) + j
                    df0.loc[ my_index, "max_horizon"] = list_max_horizon[l]
                    df0.loc[ my_index, "enter_value"] = en
                    df0.loc[ my_index, "profit_taking_param"] = li1[i]
                    df0.loc[ my_index, "stop_loss_param"] = li2[j]
        df0 = df0.set_index(["enter_value", "max_horizon", "profit_taking_param", "stop_loss_param"])
    else:
        df0 = pd.DataFrame(columns=["max_horizon","profit_taking_param", "stop_loss_param", 'profit_mean', 'profit_std','Sharpe_ratio', 'semideviation', 'Sharpe_ratio_with_semideviation', 'VaR', 'ES', 'probab_loss'])
        for l in range(len(list_max_horizon)):
            for i in range(len(li1)):
                for j in range(len(li2)):
                    my_index = l*(len(li1)*len(li2)) + i*(len(li2)) + j
                    df0.loc[ my_index, "max_horizon"] = list_max_horizon[l]
                    df0.loc[ my_index, "profit_taking_param"] = li1[i]
                    df0.loc[ my_index, "stop_loss_param"] = li2[j]
        df0 = df0.set_index(["max_horizon","profit_taking_param", "stop_loss_param"])

    del en; del li1; del li2; del my_index

    return df0

#---------------------------------------------------------------------------------------------------------------------

def save_results(df_aux, mh, en, pt, sl, results):
    '''This function saves the results of a given set of thresholds to a given row of a dataframe.'''

    if (en != None):
        for field_label in ['profit_mean', 'profit_std', 'Sharpe_ratio','semideviation', 'Sharpe_ratio_with_semideviation','VaR', 'ES', 'probab_loss']:
            df_aux.loc[(en, mh, pt, sl), field_label] = results[field_label]
    else:
        for field_label in ['profit_mean', 'profit_std', 'Sharpe_ratio','semideviation', 'Sharpe_ratio_with_semideviation','VaR', 'ES', 'probab_loss']:
            df_aux.loc[(mh,pt, sl), field_label] = results[field_label]

    del en; del pt; del sl; del results

    return df_aux

#---------------------------------------------------------------------------------------------------------------------

def find_thresholds_heatmap( input_params, rv_params, OU_params=None ):
    ''' This function finds the optimal strategy (i.e. the optimal profit-taking and stop-loss thresholds) for the given
    input parameters. It is based on the function called "batch", written by Marcos Lopez de Prado, and available in
    https://quantresearch.org/OTR.py and in Chapter 13 of his book "Advances in financial machine learning".
    Some of the input parameters are:

    :param OU_params: parameters of the Ornstein-Uhlenbeck equation: "E0" is the mean-reversion parameter, "tau" is the
    half-life parameter (which determines the speed parameter phi), and "sigma" is the standard deviation of the residuals
    of the residuals of the regression of prices. The explanations on the fitting can be found in references [1] or [2].
    The discretized Ornstein-Uhlenbeck equation is: P(t) = (1-phi)·E0 + phi·P(t-1) + sigma·epsilon, where epsilon is a
    random variable with standard normal distribution. In realistic analyses the phi, E0 and sigma parameters should be obtained
    from a linear (ols) regression of the cloud of points of P(t)-vs-P(t-1), (1-phi)·E0 being the interception, and sigma
    being the standard deviation of the residuals of the regression.
    :param input_params.list_enter_value: list of real numbers (mod E0) to enter a deal.
    :param input_params.list_profit_taking: list of real numbers which are the values to analyse for the upper threshold (profit-taking)
    :param input_params.list_stop_loss: list of real numbers which are the values to analyse for the lower threshold (stop-loss)
    :param input_params.min_num_trials: minimum number of iterations for the determination of the optimal threshold. Take a high number and check
    the convergence of the outputwrt to this parameter.
    :param input_params.max_horizon: maximum number of periods (discrete times in the Ornstein-Uhlenbeck equation above) before the position (investment) is closed.
    :return: table with the profit-taking and stop-loss thresholds and the corresponding average profit, standard devition of the profit, and
    quotient between them (:= Sharpe ratio).
    '''

    df_out = pd.DataFrame()
    min_num_trials = int(input_params.min_num_trials)
    tolerance = module_parameters.tolerance_for_heatmap

    if (input_params.evolution_type == "Ornstein-Uhlenbeck_equation"):
        for en in input_params.list_enter_value:
            gc.collect()
            adpt_list_profit_taking, adpt_list_stop_loss = adapt_lists_thresholds( input_params, en )
            df_aux = create_aux_df_heatmap( input_params.list_max_horizon, en, adpt_list_profit_taking, adpt_list_stop_loss )
            for mh in input_params.list_max_horizon:
                for pt in adpt_list_profit_taking: # The prefix "adpt_" means that the enter value (en) was added
                    for sl in adpt_list_stop_loss:
                        results = find_thresholds( mh, en, pt, sl, input_params, OU_params, rv_params, min_num_trials, tolerance )
                        df_aux = save_results(df_aux, mh, en, pt, sl, results)
            df_out=pd.concat([df_out,df_aux],axis=0)
    else: # Single product
        adpt_list_profit_taking, adpt_list_stop_loss = adapt_lists_thresholds( input_params, None )
        df_aux = create_aux_df_heatmap( input_params.list_max_horizon, None, adpt_list_profit_taking, adpt_list_stop_loss )
        for mh in input_params.list_max_horizon:
            for pt in adpt_list_profit_taking:
                for sl in adpt_list_stop_loss:
                    results = find_thresholds( mh, None, pt, sl, input_params, OU_params, rv_params, min_num_trials, tolerance )
                    df_aux = save_results(df_aux, mh, None, pt, sl, results)
        df_out=pd.concat([df_out,df_aux],axis=0)
        gc.collect()

    del tolerance; del min_num_trials; del results; del df_aux; del input_params; del OU_params; del rv_params

    return df_out

#----------------------------------------------------------------------------------------



