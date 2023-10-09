
import module_parameters
import numpy as np
from numpy.random import uniform
import pandas as pd
from scipy.stats import norm, nct
from itertools import product
from os import path
from module_generic_functions import find_thresholds
from module_plots import define_results_name,  reformat_param

# WARNING: Transaction codes and discount rates were not implemented in this module

#-----------------------------------------------------------------------------------------------------------------------

def name_file_seek_enter(input_params,OU_params,rv_params):
    '''This generates the name of the file which stores the seek for optimal enter parameters.'''

    distr = rv_params['distribution_type']

    if (input_params.path_rv_params != None):

        res_name = "OrnUhl-" + (OU_params["product_label"]).replace(".csv", "") + "-" + distr

    else:

        if (input_params.evolution_type == "Ornstein-Uhlenbeck_equation"):
            E0 = OU_params['E0'];  tau = OU_params['tau'];
            if (isinstance(E0, float)):   E0 = "{:.4f}".format(E0)
            if (isinstance(tau, float)):  tau = "{:.4f}".format(tau)
            res_name = "OrnUhl-E0_" + str(E0) + "__tau_" + str(tau) + "-" + distr
        else:
            res_name = "single-" + distr + "_"

            mu = reformat_param(rv_params['mu'])
            sigma = reformat_param(rv_params['sigma'])
            third_param = reformat_param(rv_params['third_param'])  # Skewness parameter in t-student
            fourth_param = reformat_param(rv_params['fourth_param'])  # Degrees of freedom parameter in t-student
            fifth_param = reformat_param(rv_params['fifth_param'])

            if (distr == "norm"):
                res_name += "mu_" + mu + "__sigma_" + sigma
            elif (distr == "nct"):
                res_name += "mu_" + mu + "__sigma_" + sigma + "__sk_" + third_param + "__df_" + fourth_param
            elif (distr == "genhyperbolic"):
                res_name += "mu_" + mu + "__sigma_" + sigma + "__3p_" + third_param + "__4p_" + fourth_param + "__5p_" + fifth_param
            elif (distr == "levy_stable"):
                res_name += "mu_" + mu + "__sigma_" + sigma + "__3p_" + third_param + "__4p_" + fourth_param

    del input_params; del OU_params; del rv_params

    return res_name

# -----------------------------------------------------------------------------------------------------------------------

def create_aux_df_opt_enter( list_enter_values ):
    '''This function creates an input dataframe with index corresponding to the enter values.'''

    df0 = pd.DataFrame(columns=["enter_value","profit_taking_param", "stop_loss_param","Niter", 'profit_mean', 'profit_std', 'Sharpe_ratio'])
    for i in range(len(list_enter_values)):
            df0.loc[ i, "enter_value"] =  list_enter_values[i]
    df0 = df0.set_index(["enter_value"])

    del list_enter_values;

    return df0

#---------------------------------------------------------------------------------------------------------------------

def save_results(df_aux, en, pt, sl, Niter, quantity, value ):
    '''This function saves the results of a given set of thresholds to a given row of a dataframe.'''

    df_aux.loc[en, "profit_taking_param"] = pt
    df_aux.loc[en, "stop_loss_param"] = sl
    df_aux.loc[en, "Niter"] = Niter
    df_aux.loc[en, quantity] = value

    del en; del pt; del sl; del Niter; del quantity; del value

    return df_aux

#---------------------------------------------------------------------------------------------------------------------

def find_optimum_of_file( path_results_file, quantity="Sharpe_ratio" ):
    '''This function finds the profit-taking and stop-loss thresholds which provide best results. This can be used as
    initial guess for the gradient-descent calculation of the optimal strategy.'''

    if not (path.exists(path_results_file)): return None, None, None

    df0 = pd.read_csv( path_results_file, header=0, usecols=["enter_value","profit_taking_param","stop_loss_param",quantity] )

    if (quantity in ["profit_mean", "Sharpe_ratio", "Sharpe_ratio_with_semideviation"]):
        opt_quantity = max( df0[quantity] )
    elif (quantity in ["profit_std","semideviation", "VaR", "ES", "probab_loss"]):
        opt_quantity = min( df0[quantity])
    else:
        raise Exception("\nERROR: Unknown quantity",quantity,"in",path_results_file)

    df0.set_index(quantity,inplace=True)
    df_aux = df0.loc[opt_quantity, "enter_value"]
    if ( isinstance(df_aux,float) or isinstance(df_aux,int) ): en = df_aux
    else: en = df_aux.iat[0]
    df_aux = df0.loc[opt_quantity, "profit_taking_param"]
    if (isinstance(df_aux,float) or isinstance(df_aux,int) ): pt = df_aux
    else: pt = df_aux.iat[0]
    df_aux = df0.loc[opt_quantity, "stop_loss_param"]
    if (isinstance(df_aux,float) or isinstance(df_aux,int)  ): sl = df_aux
    else: sl = df_aux.iat[0]

    del df0; del quantity; del opt_quantity

    return en, pt, sl

#-----------------------------------------------------------------------------------------------------------------------

def threshold_plus_delta( thrsh_in, enter_value, type_param, list_params ):
    '''This function adds a small delta to a profit-taking or stop-los threshold'''

    assert type_param in ["PT","SL"]

    if (type_param=="PT"):
        if (len(list_params)>1):
            thrsh_out = thrsh_in +(np.sign(enter_value))* abs(list_params[1] - list_params[0])/8
        else:
            thrsh_out = thrsh_in *(1+np.sign(enter_value)*0.05)
    else:
        if (len(list_params)>1):
            thrsh_out = thrsh_in - (np.sign(enter_value))* abs(list_params[1] - list_params[0])/8
        else:
            thrsh_out = thrsh_in *(1-np.sign(enter_value)*0.05)

    del thrsh_in; del enter_value; del type_param; del list_params

    return thrsh_out

#-----------------------------------------------------------------------------------------------------------------------

def define_start_points(  input_params, OU_params, rv_params, enter_value, pt1=None, sl1=None  ):
    '''This function finds a start guess for the calculation of the optimal pair of thresholds.
    It indeed provides the positions of the first two iterations to be used in the Barzilai-Borwein optimization.'''

    # Using the exit thresholds from a different enter_value as start point
    if ((pt1!=None) and (sl1!=None)):
        pt0 = threshold_plus_delta(pt1, enter_value, "PT", input_params.list_profit_taking)
        sl0 = threshold_plus_delta(sl1, enter_value, "SL", input_params.list_stop_loss)
        del input_params; del OU_params; del rv_params; del enter_value
        return pt0, sl0, pt1, sl1

    # Reading the start point from heatmap
    res_name = define_results_name(input_params, OU_params, rv_params)
    trash, pt1, sl1 = find_optimum_of_file( f'{input_params.output_directory}/Results/{res_name}.csv', input_params.quantity_to_analyse )

    if ( (pt1==None) or (sl1==None)):
        print("\nSEVERE WARNING: No results from a previous heatmap have been found for \n",OU_params,"\n", rv_params)
        print("Therefore the seek for the optimum thresholds will likely fall to a local minimum of the "+input_params.quantity_to_analyse+".")
        print("We strongly suggest you to run a calculation with <<output_type='heat-map'>> in input.py (and fine grid)")
        print("before you run your calculation with  <<output_type='optimal_solution'>>.\n")
        ''' 
        # For gradient descent
        pt0 = ( input_params.list_profit_taking[0] + input_params.list_profit_taking[-1] ) / 2
        sl0 = (input_params.list_stop_loss[0] + input_params.list_stop_loss[-1]) / 2
        grad0_pt, grad0_sl = find_gradient( 0,input_params.quantity_to_analyse, pt0, sl0, input_params, OU_params, rv_params, input_params.min_num_trials, 0.00001, 0)
        pt1 = pt0 + (0.5+grad0_pt) * abs(input_params.list_profit_taking[1] - input_params.list_profit_taking[0])/8
        sl1 = sl0 + (0.5+grad0_sl) * abs( input_params.list_stop_loss[1] - input_params.list_stop_loss[0])/8
        '''
        # For bisection
        pt0 = enter_value+min( input_params.list_profit_taking[0], input_params.list_profit_taking[-1] )
        pt1 = enter_value+max(input_params.list_profit_taking[0], input_params.list_profit_taking[-1])
        sl0 = enter_value+min( input_params.list_stop_loss[0], input_params.list_stop_loss[-1])
        sl1 = enter_value+max(input_params.list_stop_loss[0], input_params.list_stop_loss[-1])
    else:
        pt0 = threshold_plus_delta(pt1, enter_value, "PT", input_params.list_profit_taking)
        sl0 = threshold_plus_delta(sl1, enter_value, "SL", input_params.list_stop_loss)

    del res_name; del input_params; del OU_params; del rv_params; del enter_value

    return pt0, sl0, pt1, sl1

#-----------------------------------------------------------------------------------------------------------------------

def find_gradient( iter, quantity, mh, pt, sl, input_params, OU_params, rv_params, min_num_trials, tolerance, enter_value ):
    ''' This function calculates the gradient of a given "quantity" (e.g. Sharpe ratio) with respect to the variables
    profit-taking and stop-loss thresholds.

    :param quantity:
    :param pt:
    :param sl:
    :param input_params:
    :param OU_params:
    :param rv_params:
    :param min_num_trials:
    :param tolerance:
    :param seed:
    :return:
    '''

    delta_pt = 0.05 ; delta_sl = 0.05 #xx

    results0 = find_thresholds( mh, enter_value, pt-delta_pt, sl, input_params, OU_params, rv_params, min_num_trials, tolerance )
    quantity_to_optimize0    = results0[quantity] #find_quantity_to_optimize( quantity, results )

    results1 = find_thresholds( mh, enter_value, pt+delta_pt, sl, input_params, OU_params, rv_params, min_num_trials, tolerance )
    quantity_to_optimize1 = results1[quantity] #find_quantity_to_optimize( quantity, profit_avg1, profit_std1  )

    results2 = find_thresholds( mh, enter_value, pt, sl-delta_sl, input_params, OU_params, rv_params, min_num_trials, tolerance )
    quantity_to_optimize2 = results2[quantity] #find_quantity_to_optimize( quantity, profit_avg2, profit_std2  )

    results3 = find_thresholds( mh, enter_value, pt, sl+delta_sl, input_params, OU_params, rv_params, min_num_trials,tolerance )
    quantity_to_optimize3 = results3[quantity] #find_quantity_to_optimize(quantity, profit_avg3, profit_std3)

    grad_pt = (quantity_to_optimize1-quantity_to_optimize0)/(2*delta_pt)
    grad_sl = (quantity_to_optimize3-quantity_to_optimize2)/(2*delta_sl)

    if (abs(grad_pt) < 0.00000001 ): grad_pt = abs(threshold_plus_delta(pt, enter_value, "PT", input_params.list_profit_taking)-pt) * uniform(-1,1)/(iter+2)
    if (abs(grad_sl) < 0.00000001 ): grad_sl = abs(threshold_plus_delta(sl, enter_value, "SL", input_params.list_stop_loss)-sl) * uniform(-1,1)/(iter+2)

    if (input_params.verbose==2): print(pt, sl, quantity_to_optimize0, "Grad=", grad_pt, grad_sl)

    del delta_sl; del delta_pt; del results0; del quantity_to_optimize0; del results1; del quantity_to_optimize1; del results2; del quantity_to_optimize2; del results3; del quantity_to_optimize3;

    return grad_pt, grad_sl

#-----------------------------------------------------------------------------------------------------------------------

def find_optimal_thresholds(input_params, rv_params, OU_params ):
    '''This function is a wrapper that calls the corresponding method to find the optimal solution. It sweeps the enter and exit
    values for the pairs trading, yet it is intended to provide the optimal enter value. After having found the optimal enter value,
    a calculation with output_type = heatmap is recommended. But it is also recommended to run a heatmap calculation before
    running this equation, because it will provide an appropriate starting guess for the profit-taking and stop-loss thresholds.
    Note that it is frequent that the Sharpe ratio falls to a local minimum if output_type = "optimal_solution"  '''

    print("The optimal solution will be calculaded using the "+module_parameters.dict_text_opt_method[input_params.method_optimal_solution]+" method.")
    #print("The tried enter values are:",input_params.list_enter_value)
    print("For each analysed enter value:")
    print(" * The minimum number of iterations is " + str(module_parameters.min_iter_search_optimum) + "; the maximum number of iterations is " + str(module_parameters.max_iter_search_optimum) + ".")
    print(" * The calculations will iterate until the relative difference between two consecutive values of " + input_params.quantity_to_analyse + " is less than " + str(module_parameters.tolerance_for_optimum) + ".")
    if (input_params.evolution_type == "Ornstein-Uhlenbeck"): print("The parameters of the Ornstein-Uhlenbeck equation are", OU_params)
    print("The parameters of the random variable are", rv_params,"\n")

    if (not input_params.method_optimal_solution in ["Barzilai-Borwein", "bisection"]):
        raise Exception("\nERROR: The method to calculate the optimal solution must be either 'Barzilai-Borwein' or 'bisection' (not" + str(input_params.method_optimal_solution) + ").\n")

    pt_start = 0.0; sl_start=-18.0 # pt_start=None; sl_start=None
    df_out = create_aux_df_opt_enter(input_params.list_enter_value)

    for enter_value in input_params.list_enter_value:
        if ( input_params.method_optimal_solution=="Barzilai-Borwein" ):
            pt_start, sl_start, df_out = find_optimal_exit_thresholds_bb(input_params, OU_params, rv_params, mh, enter_value, pt_start, sl_start, df_out )
        else:
            pt_start, sl_start, df_out = find_optimal_exit_thresholds_bisection(input_params, OU_params, rv_params, mh, enter_value, pt_start, sl_start, df_out )

    print("\n\n====>> The seek for oprimal thresholds concluded. Its results are:\n")
    pd.set_option('max_columns', 20); pd.set_option('max_rows', 99999)
    print( df_out[ ["profit_taking_param","stop_loss_param","Niter",input_params.quantity_to_analyse] ] )
    filepath_results = input_params.output_directory+"/Results/Optimal_enter_"+input_params.method_optimal_solution+"_"+name_file_seek_enter(input_params,OU_params,rv_params)+".csv"
    df_out.reset_index(inplace=True)
    df_out.to_csv( filepath_results,index=False)
    enter_value, pt, sl = find_optimum_of_file(filepath_results,input_params.quantity_to_analyse)

    print("\n******  The value to enter the positions which maximizes the "+input_params.quantity_to_analyse+" is " + "{:.6f}".format(enter_value)+"  *****")
    print("       The corresponding profit-taking and stop-loss parameters are: " +"{:.4f}".format(pt)+", {:.4f}".format(sl)+"\n" )

    del input_params; del OU_params; del rv_params; del df_out; del filepath_results; del enter_value; del pt; del sl

    return

#-----------------------------------------------------------------------------------------------------------------------

def find_optimal_exit_thresholds_bb( input_params, OU_params, rv_params, mh, enter_value, pt_start, sl_start, df_results ):
    '''This function finds the optimal value of the pair of thresholds (profit-taking and stop-loss) using the
    gradient descent method (Barzilai-Borwein variant).'''

    quantity_to_opt = input_params.quantity_to_analyse
    min_num_trials = int(input_params.min_num_trials)
    tol = module_parameters.tolerance_for_optimum

    pt0, sl0, pt1, sl1 = define_start_points(input_params, OU_params, rv_params, enter_value, pt_start, sl_start )
    grad0_pt, grad0_sl = find_gradient(0,quantity_to_opt, mh, pt0, sl0, input_params, OU_params, rv_params, min_num_trials, tol, enter_value)
    grad1_pt, grad1_sl = find_gradient(0,quantity_to_opt, mh, pt1, sl1, input_params, OU_params, rv_params, min_num_trials, tol, enter_value )

    array_check_convergence = [-999, -999, 999, 999]

    for iter in range(module_parameters.max_iter_search_optimum):

        gradFdiff = [grad1_pt - grad0_pt, grad1_sl - grad0_sl]
        den = (gradFdiff[0] * gradFdiff[0] + gradFdiff[1] * gradFdiff[1])
        if (den < 0.00000001): break
        gamma_bb = -(gradFdiff[0] * (pt1 - pt0) + gradFdiff[1] * (sl1 - sl0)) / den
        if(input_params.verbose==2): print("gamma",gamma_bb,"\n")
        if (input_params.quantity_to_analyse in ["standard_deviation","semideviation", "VaR", "ES", "probab_loss"]): gamma_bb *= (-1)

        pt0 = pt1;           sl0 = sl1
        grad0_pt = grad1_pt; grad0_sl = grad1_sl

        pt1 += gamma_bb * (grad1_pt); sl1 += gamma_bb * (grad1_sl)

        # Capping the value of the thresholds to set limits
        if (enter_value<=0):
            pt1 = max( pt1, enter_value)
            pt1 = min( pt1, enter_value+max(abs(input_params.list_profit_taking[-1]),abs(input_params.list_profit_taking[0])))
            sl1 = min( sl1, enter_value)
            sl1 = max( sl1, enter_value-max(abs(input_params.list_stop_loss[-1]),abs(input_params.list_stop_loss[0])) )
        else:
            pt1 = min(pt1, enter_value)
            pt1 = max(pt1, enter_value - max(abs(input_params.list_profit_taking[-1]),abs(input_params.list_profit_taking[0])))
            sl1 = max(sl1, enter_value)
            sl1 = min(sl1, enter_value + max(abs(input_params.list_stop_loss[-1]), abs(input_params.list_stop_loss[0])))

        grad1_pt, grad1_sl = find_gradient(iter,quantity_to_opt, mh, pt1, sl1, input_params, OU_params, rv_params,min_num_trials, tol, enter_value )

        #if ( (iter%3)==0):
        #    array_check_convergence[ 2*(round(iter/3)%2)   ] = pt1
        #    array_check_convergence[ 2*(round(iter/3)%2)+1 ] = sl1
        #    change_sol = abs( array_check_convergence[0]-array_check_convergence[2] ) + abs(array_check_convergence[1]-array_check_convergence[3])
        #    if ( (iter > module_parameters.min_iter_search_optimum) and ( change_sol < module_parameters.tolerance_search_optimum ) ): break

        # We check that the relative change in e.g. the Sharpe ratio between two consecutive iterations is below the specified tolerance
        results1 = find_thresholds( mh, enter_value, pt1, sl1, input_params, OU_params, rv_params, min_num_trials, tol)
        quantity_to_optimize1 = results1[quantity_to_opt] # find_quantity_to_optimize(input_params.quantity_to_analyse, profit_avg1, profit_std1)
        array_check_convergence[iter%2] = quantity_to_optimize1
        change_sol = abs( (array_check_convergence[1]-array_check_convergence[0])/array_check_convergence[abs(iter%2-1)] )
        if ((iter > module_parameters.min_iter_search_optimum) and (change_sol < module_parameters.tolerance_search_optimum)): break

    print("\n*** Enter value = "+"{:.6f}".format(enter_value)+" ***\nWe found the optimum after",iter+1,"iterations; It is: PT=","{:.5f}".format(pt1),"; SL=","{:.5f}".format( sl1) )
    results1 = find_thresholds( mh, enter_value,pt1, sl1, input_params, OU_params, rv_params, min_num_trials*5, tol)
    print("Its corresponding results are: avg_profit=","{:.4f}".format(results1["profit_mean"]),"; standard dev. of profit=","{:.4f}".format(results1["profit_std"]),"; "+quantity_to_opt+ " = ","{:.5f}".format(results1[quantity_to_opt]) )

    df_results = save_results(df_results, enter_value, pt1, sl1, iter, quantity_to_opt, results1[quantity_to_opt])

    del results1; del quantity_to_optimize1; del change_sol; del array_check_convergence
    del pt0; del sl0; del grad0_pt; del grad1_pt; del grad0_sl; del grad1_sl

    return pt1, sl1, df_results

# -----------------------------------------------------------------------------------------------------------------------

def bisection_pt(pt0, sl, pt1, input_params, OU_params, rv_params, min_num_trials, tol, enter_value):

    quantity_to_optimize0 = 999;
    quantity_to_optimize1 = -999
    delt = abs( input_params.list_profit_taking[-1] - input_params.list_profit_taking[0] ) / len(input_params.list_profit_taking)
    if (pt0==None):
        pt0 = pt1 - delt/4
        pt1 = pt1 + delt/4
    iter = 0
    while ((iter < 20) and (abs(quantity_to_optimize1 - quantity_to_optimize0) > 0.0001)):

        results0 = find_thresholds( mh, enter_value,pt0, sl, input_params, OU_params, rv_params, min_num_trials, tol)
        quantity_to_optimize0 = results0[input_params.quantity_to_analyse]# find_quantity_to_optimize(input_params.quantity_to_analyse, profit_avg0, profit_std0)
        results1 = find_thresholds( mh, enter_value,pt1, sl, input_params, OU_params, rv_params, min_num_trials, tol)
        quantity_to_optimize1 = results1[input_params.quantity_to_analyse] #find_quantity_to_optimize(input_params.quantity_to_analyse, profit_avg1, profit_std1)
        delta = (pt1 - pt0) / 2

        if(input_params.verbose==2): print(pt0, quantity_to_optimize0, ";", pt1, quantity_to_optimize1)

        if (input_params.quantity_to_analyse in ["Sharpe_ratio", "average", "Sharpe_ratio_with_semideviation"]):
            if (quantity_to_optimize0 > quantity_to_optimize1):
                pt1 = pt0
                pt0 -= delta
            else:
                pt0 = pt1
                pt1 += delta
        elif (input_params.quantity_to_analyse in ["standard_deviation","semideviation", "VaR", "ES", "probab_loss"]):
            if (quantity_to_optimize0 < quantity_to_optimize1):
                pt1 = pt0
                pt0 -= delta
            else:
                pt0 = pt1
                pt1 += delta

    if (enter_value<=0):
        lim = max(abs(input_params.list_profit_taking[-1]),abs(input_params.list_profit_taking[0]))
        if (pt1 > enter_value+lim ): return  enter_value+lim
        if (pt0 < enter_value  ): return  enter_value
    else:
        lim = min( -abs(input_params.list_profit_taking[-1]), -abs(input_params.list_profit_taking[0]))
        if (pt1 < enter_value+lim ): return  enter_value+lim
        if (pt0 > enter_value   ): return  enter_value

    del delta; del delt; del results0; del quantity_to_optimize0; del results1; del quantity_to_optimize1; del enter_value

    return (pt0 + pt1) / 2

# -----------------------------------------------------------------------------------------------------------------------

def bisection_sl(pt, sl0, sl1, input_params, OU_params, rv_params, min_num_trials, tol, enter_value):

    quantity_to_optimize0 = 999
    quantity_to_optimize1 = -999
    delt = abs(input_params.list_stop_loss[-1] - input_params.list_stop_loss[0]) / len(input_params.list_stop_loss)
    if (sl1==None):
        sl0=sl0-delt/4
        sl1=sl0+delt/4
    iter = 0

    while ( (iter < 20) and (abs(quantity_to_optimize1 - quantity_to_optimize0) > 0.0001)):

        results0 = find_thresholds( mh, enter_value, pt, sl0, input_params, OU_params, rv_params, min_num_trials, tol )
        quantity_to_optimize0 = results0[input_params.quantity_to_analyse] #find_quantity_to_optimize(input_params.quantity_to_analyse, profit_avg0, profit_std0)
        results1 = find_thresholds( mh, enter_value, pt, sl1, input_params, OU_params, rv_params, min_num_trials, tol )
        quantity_to_optimize1 = results1[input_params.quantity_to_analyse] # find_quantity_to_optimize(input_params.quantity_to_analyse, profit_avg1, profit_std1)
        delta = (sl1 - sl0) / 2
        if(input_params.verbose==2): print(sl0, quantity_to_optimize0, ";", sl1, quantity_to_optimize1)

        if (input_params.quantity_to_analyse in ["Sharpe_ratio", "average"]):
            if (quantity_to_optimize0 > quantity_to_optimize1):
                sl1 = sl0
                sl0 -= delta
            else:
                sl0 = sl1
                sl1 += delta
        elif (input_params.quantity_to_analyse == "standard_deviation"):
            if (quantity_to_optimize0 < quantity_to_optimize1):
                sl1 = sl0
                sl0 -= delta
            else:
                sl0 = sl1
                sl1 += delta

    if (enter_value <= 0):
        lim = min(-abs(input_params.list_stop_loss[-1]), -abs(input_params.list_stop_loss[0]))
        if (sl1 > enter_value ): return enter_value
        if (sl0 < enter_value+lim): return enter_value+lim
    else:
        lim = max(abs(input_params.list_stop_loss[-1]), abs(input_params.list_stop_loss[0]))
        if (sl1 > enter_value + lim): return enter_value + lim
        if (sl0 < enter_value): return enter_value

    del delta; del delt; del results0; del quantity_to_optimize0; del results1; del quantity_to_optimize1;

    return (sl0 + sl1) / 2

# -----------------------------------------------------------------------------------------------------------------------

def update_best_so_far(best_so_far, pt, sl, quantity_to_optimize, quantity):
    '''This function stores the thesholds (pt, sl) with the best quantity (e.g. Sharpe ratio) found so far.'''
    if  (quantity=="standard_deviation"):
        if (quantity_to_optimize < best_so_far["quantity"] ):
            best_so_far["pt"]=pt
            best_so_far["sl"]=sl
            best_so_far["quantity"]=quantity_to_optimize
    else: # Sharpe ratio or average profit
        if ( quantity_to_optimize > best_so_far["quantity"] ):
            #print("\nWe have updated the best-so-far: It was pt=", best_so_far["pt"], "SR=", best_so_far["quantity"])
            best_so_far["pt"]=pt
            best_so_far["sl"]=sl
            best_so_far["quantity"]=quantity_to_optimize
            #print(" Now it is pt=", best_so_far["pt"], "SR=", best_so_far["quantity"])
    return best_so_far

# -----------------------------------------------------------------------------------------------------------------------

def find_optimal_exit_thresholds_bisection(input_params, OU_params, rv_params, mh, enter_value, pt_start, sl_start, df_results ):
    '''This function finds the optimal value of the pair of thresholds (profit-taking and stop-loss) using the
    gradient descent method.
    WARNING: This function is much slower than find_optimal_exit_thresholds_bb, it should be used only for checking purposes.'''

    quantity_to_opt = input_params.quantity_to_analyse
    min_num_trials = int(input_params.min_num_trials)
    tol = module_parameters.tolerance_for_optimum
    best_so_far = {"pt":999, "sl":-999, "quantity":0}

    pt0, sl0, pt1, sl1 = define_start_points(input_params, OU_params, rv_params, enter_value, pt_start, sl_start )

    #print("ZZ Prepre", pt0, sl0,";", pt1, sl1)

    array_check_convergence = [999,-999]
    pt1 = bisection_pt(pt0, sl0, pt1, input_params, OU_params, rv_params, min_num_trials, tol, enter_value)
    #print("ZZ Pre A", pt1, sl0)
    sl0 = bisection_sl(pt1, sl0, sl1, input_params, OU_params, rv_params, min_num_trials, tol, enter_value)

    #print("ZZ Pre B",pt1,sl0)

    for iter in range(module_parameters.max_iter_search_optimum):

        pt1 = bisection_pt(None, sl0, pt1, input_params, OU_params, rv_params, min_num_trials, tol, enter_value)
        if(input_params.verbose==2): print("--")
        sl0 = bisection_sl(pt1, sl0, None, input_params, OU_params, rv_params, min_num_trials, tol, enter_value)
        if(input_params.verbose==2): print()

        # We check that the relative change in e.g. the Sharpe ration between two consecutive iterations is below the specified tolerance
        results1 = find_thresholds( mh, enter_value, pt1, sl0, input_params, OU_params, rv_params, min_num_trials, tol)
        quantity_to_optimize = results1[quantity_to_opt]
        array_check_convergence[iter%2]== quantity_to_optimize
        best_so_far = update_best_so_far(best_so_far, pt1, sl0, quantity_to_optimize, quantity_to_opt)
        change_sol = abs( (array_check_convergence[1]-array_check_convergence[0])/array_check_convergence[abs(iter%2-1)] )
        if ((iter > module_parameters.min_iter_search_optimum) and (change_sol < module_parameters.tolerance_search_optimum)): break

    print("\n*** Enter value = " + "{:.6f}".format(enter_value) + "***\nWe found the optimum after", iter+1, "iterations; It is: PT=", "{:.5f}".format(best_so_far["pt"]), "; SL=","{:.5f}".format(best_so_far["sl"]) )

    results1 = find_thresholds( mh, enter_value, best_so_far["pt"], best_so_far["sl"], input_params, OU_params, rv_params,min_num_trials * 5, tol )
    print("Its corresponding results are: avg_profit=", "{:.4f}".format(results1["profit_mean"]),"; standard dev. of profit=", "{:.4f}".format(results1["profit_std"]), "; Sharpe ratio = ", "{:.5f}".format(results1["Sharpe_ratio"] ))

    df_results = save_results(df_results, enter_value, pt1, sl1, iter, quantity_to_opt, results1[quantity_to_opt])

    if (input_params.verbose == 2):
        if (  (abs(best_so_far["pt"]-(enter_value+input_params.list_profit_taking[0])) < 0.000001 ) or  (abs(best_so_far["pt"]-(enter_value+input_params.list_profit_taking[-1])) < 0.000001 ) ):
            print("\nWARNING: The profit-taking parameter is at the limit of the allowed values ("+str(best_so_far["pt"])+").\n")
        if ((abs(best_so_far["sl"] - (enter_value+input_params.list_stop_loss[0])) < 0.000001) or (abs(best_so_far["pt"] - (enter_value+input_params.list_stop_loss[-1])) < 0.000001)):
            print("\nWARNING: The stop-loss parameter is at the limit of the allowed values (" + str(best_so_far["sl"]) + ").\n")

    del pt0; del sl0; del pt1; del sl1; del min_num_trials; del tol; del quantity_to_opt; del enter_value; del array_check_convergence; del results1; del quantity_to_optimize

    return best_so_far["pt"], best_so_far["sl"], df_results

#-----------------------------------------------------------------------------------------------------------------------
