
'''This module contains functions which are called by different kinds of calculations (e.g. heatmap or optimal-solution,
Ornstein-Uhlenbeck or single-product).'''

import numpy as np
from numpy.random import uniform
from scipy.stats import norm, nct, genhyperbolic, levy_stable, uniform #from random import gauss
# import pandas as pd
import module_parameters


#----------------------------------------------------------------------------------------

def find_quantity_to_optimize( quantity, profit_mean, profit_std  ):
    if (quantity=="Sharpe_ratio"): return profit_mean/profit_std
    if (quantity=="profit_mean"):   return profit_mean
    if (quantity=="profit_std"):   return profit_std
    raise Exception("\nERROR: The quantity to optimize "+str(quantity)+" must be in: [profit_mean, profit_std, Sharpe_ratio].")

#---------------------------------------------------------------------------------------------------------------------

def update_discount_factor( DF, iteration, discount_factor_unit_time, discount_factor_update_method ):
    '''This function updates the discount factor'''

    if ( (discount_factor_unit_time==0) or (discount_factor_unit_time==None) ):
        return 1
    if (discount_factor_update_method=="homogeneous"):
        #print(iteration, ") in=",DF,"; out=",DF * discount_factor_unit_time)
        return DF * discount_factor_unit_time
    elif (discount_factor_update_method=="weekly"):
        if ( ((iteration+1)%5 )==0 ):
            return DF * discount_factor_unit_time * discount_factor_unit_time * discount_factor_unit_time
        else:
            return DF * discount_factor_unit_time
    else:
        raise Exception("\nERROR: Unrecognized disctounting periodicity "+str(discount_factor_update_method)+"\n")

#---------------------------------------------------------------------------------------------------------------------

def adapt_lists_thresholds(input_params, enter_value):
    '''This function rewrites the lists of profit-taking and stop-loss possible values.'''

    # We avoid that enter_value=0 on a list of NEGATIVE enter values is considered a positive enter value
    if (abs(enter_value)<0.0000000001):
        enter_value = 0.0000000000001 * np.sign( np.mean( ( np.array( input_params.list_enter_value ) )  ) )

    if ( input_params.evolution_type =="Ornstein-Uhlenbeck_equation"):
        if (enter_value < 0):
            adapted_list_profit_taking = []
            for thresh in input_params.list_profit_taking:
                adapted_list_profit_taking.append( enter_value+abs(thresh) )
            adapted_list_profit_taking.sort()
            adapted_list_stop_loss = []
            for thresh in input_params.list_stop_loss:
                adapted_list_stop_loss.append(enter_value - abs(thresh) )
            adapted_list_stop_loss.sort() #reverse=True
        elif (enter_value > 0): # enter_value>0
            adapted_list_profit_taking = []
            for thresh in input_params.list_profit_taking:
                adapted_list_profit_taking.append(enter_value - abs(thresh))
            adapted_list_profit_taking.sort()
            adapted_list_stop_loss = []
            for thresh in input_params.list_stop_loss:
                adapted_list_stop_loss.append(enter_value + abs(thresh))
            adapted_list_stop_loss.sort()
    else: #input_params.evolution_type is "single_product"
        adapted_list_profit_taking = []
        for thresh in input_params.list_profit_taking:
            adapted_list_profit_taking.append(1+abs(thresh))
        adapted_list_profit_taking.sort()
        adapted_list_stop_loss = []
        for thresh in input_params.list_stop_loss:
            adapted_list_stop_loss.append( 1 - abs(thresh))
        adapted_list_stop_loss.sort()

    #print("en=",enter_value,"\n adapted_list_profit_taking",adapted_list_profit_taking, "\nadapted_list_stop_loss", adapted_list_stop_loss)

    del input_params; del enter_value

    return adapted_list_profit_taking, adapted_list_stop_loss

#----------------------------------------------------------------------------------------

def generate_random_variable( evolution_type, rv_params, array_size, seed_factor ):
    '''This function generates a random variable (real number) with a distribution given by the user-defined input parameters.'''

    np.random.seed( seed=( round(seed_factor) ) )

    distr = rv_params['distribution_type']

    if ( ( evolution_type == "single_product") and ( distr == "norm") ): # We use the closed form of geometric Brownian motion, not its discretized equation
        return norm.rvs(size=array_size, loc=rv_params['mu'] - ((rv_params['sigma']**2)/2), scale=rv_params['sigma']) # To make it consistent with the closef form of geometric Brownian motion: $p(t)  =   p(t-1) · exp[\Delta W'] , with \Delta W' \sim N(\mu-\sigma^2/2,\sigma)$.

    if ( distr == "norm"):
        return norm.rvs(size=array_size, loc=rv_params['mu'], scale=rv_params['sigma']  ) # MLdP (slow): rv_params['sigma'] * gauss(0, 1)
    elif ( distr == "nct"):
        return nct.rvs(size=array_size, loc=rv_params['mu'], scale=rv_params['sigma'], nc=rv_params['third_param'], df=rv_params['fourth_param'] )
    elif ( distr == "genhyperbolic"):
        return genhyperbolic.rvs(size=array_size, loc=rv_params['mu'], scale=rv_params['sigma'], b=rv_params['third_param'], a=rv_params['fourth_param'], p=rv_params['fifth_param'] )
    elif ( distr == "levy_stable" ):
        return levy_stable.rvs(size=array_size, loc=rv_params['mu'], scale=rv_params['sigma'], beta=rv_params['third_param'], alpha=rv_params['fourth_param'])
    else:
        raise Exception("\nERROR: Unknown distribution " + distr + "\n")

#----------------------------------------------------------------------------------------

def find_probab_loss( array_in, len_array_in ):
    '''This function finds the index of the first entry of a sorted array which is positive, and uses it to find the probability of loss.'''

    if (array_in[-1]<0): return 1
    if (array_in[0]>0):  return 0
    left_index=0; right_index=len(array_in-1)
    while ((right_index-left_index)!=1):
        middle_index=int((left_index+right_index)/2) # middle_index is the first index with positive value of the array
        if ( array_in[middle_index]<0):left_index=middle_index
        else: right_index=middle_index

    del array_in; del right_index; del middle_index

    return (left_index+1)/len_array_in

#----------------------------------------------------------------------------------------

def calc_risks( arr_in, len_array, arr_in_mean ):
    '''This function calculates the semideviation (downside risk), Value-at-Risk (VaR) and Expected Shortfall.
    The input array must be sorted.
    The semideviation formula is the one given by Grinold/Kahn <<Active Portfolio Management>> (1999), p.45.
    The ES formula is extracted from the official European Banking Authority document
    <<EBA/RTS/2020/12, 17 December 2020, Final Draft RTS on the calculation of the
    stress scenario risk measure under Article 325bk(3) of Regulation (EU) No 575/2013 (Capital Requirements Regulation 2 – CRR2)>>
    (with changed sign).'''

    # Calculation of semi-deviation
    semideviation = 0
    for i in range(len_array):
        if (arr_in[i]>=arr_in_mean): break
        semideviation += (arr_in[i]-arr_in_mean)**2
    semideviation = np.sqrt( semideviation/len_array)

    # Calculation of VaR
    value_at_risk = arr_in[ max( round( module_parameters.alpha_for_VaR_and_ES * len_array ) -1, 0 ) ]

    # Calculation of expected shortfall
    alphaN = module_parameters.alpha_for_VaR_and_ES * len_array
    int_alphaN = int(alphaN)
    expected_shortfall = ( np.sum( arr_in[0:int_alphaN] ) + (alphaN-int_alphaN)*arr_in[int_alphaN] )*(1/alphaN)

    del arr_in; del len_array; del arr_in_mean; del i; del int_alphaN; del alphaN

    return semideviation, value_at_risk, expected_shortfall

#----------------------------------------------------------------------------------------

def find_measures(arr_profit):
    '''This function calculates the measures for a set of input results. This includes average profit, standard deviation
    of the profit, Sharpe ratio, semivariance, probability of loss, VaR and Expected Shortfall.'''

    len_array = len(arr_profit)
    if ( len_array  == 0): return  {'profit_mean': 0, 'profit_std': 0, 'Sharpe_ratio': 0, 'semideviation': 0, 'Sharpe_ratio_with_semideviation': 0, 'VaR': 0, 'ES': 0, 'probab_loss': 0}
    arr_profit   = np.sort( arr_profit )
    profit_mean  = np.mean(arr_profit)
    profit_stdev = np.std(arr_profit)
    semideviation, value_at_risk, expected_shortfall  = calc_risks( arr_profit, len_array, profit_mean  )
    probab_loss = find_probab_loss(arr_profit, len_array)
    if (abs(profit_stdev)>0.00000001): sr = profit_mean/profit_stdev
    else: sr = 0
    if (abs(profit_stdev)>0.00000001): srd = profit_mean / semideviation
    else: srd=0

    del arr_profit; del len_array

    return {'profit_mean': profit_mean, 'profit_std': profit_stdev, 'Sharpe_ratio':sr, 'semideviation':semideviation, 'Sharpe_ratio_with_semideviation':srd, 'VaR':value_at_risk, 'ES':expected_shortfall, 'probab_loss':probab_loss }

#----------------------------------------------------------------------------------------

def find_thresholds( mh, enter_value, pt, sl, input_params, OU_params, rv_params, min_num_trials, tolerance ):

    if (input_params.evolution_type == "Ornstein-Uhlenbeck_equation"):

        if (input_params.method_for_calculation_of_profits == "enter_ensured"):
            results = find_thresholds_ensured_enter_OU( mh, enter_value, pt, sl, input_params, OU_params, rv_params, min_num_trials, tolerance )
        else: # method_for_calculation_of_profits is "enter_random_one_deal", "enter_random_many_deals"
            results = find_thresholds_random_enter_OU( mh, enter_value, pt, sl, input_params, OU_params, rv_params, min_num_trials, tolerance )

    else: # input_params.evolution_type is "single_product"

        results = find_thresholds_single( mh, pt, sl, input_params, rv_params, min_num_trials, tolerance )

    del pt; del sl; del input_params; del OU_params; del rv_params; del min_num_trials; del tolerance; del enter_value

    return results

#----------------------------------------------------------------------------------------

def find_thresholds_random_enter_OU( horizon, enter_value, pt, sl, input_params, OU_params, rv_params, min_num_trials, tolerance ):
    ''' This function finds the optimal strategy (i.e. the optimal profit-taking and stop-loss thresholds) for the given
    input parameters. It is based on the function called "batch", written by Marcos Lopez de Prado, and available in
    https://quantresearch.org/OTR.py.txt and in Chapter 13 of his book "Advances in financial machine learning".
    This function calculates the profit as the additive variation of spread in positions entered during a period (e.g. one year)
    specified by the user. It repeats this process for many iterations. In each of these periods, the starting value of
    the spread is assumed to be its mean (in an Ornstein-Uhlenbeck process the mean the spread tends to).

    :param pt, sl: (floats) profit-taking and stop-loss thresholds to analyse.
    :param input_params: (class) User-defined input parameters for the calculation
    :param OU_params: (class containing floats) Parameters of the Ornstein-Uhlenbeck equation: "E0" is the mean-reversion parameter, "tau" is the
    half-life parameter (which determines the speed parameter phi), and "sigma" is the standard deviation of the residuals
    of the residuals of the regression of prices. The explanations on the fitting can be found in references [1] or [2].
    The discretized Ornstein-Uhlenbeck equation is: P(t) = (1-phi)·E0 + phi·P(t-1) + sigma·epsilon, where epsilon is a
    random variable with standard normal distribution. In realistic analyses the phi, E0 and sigma parameters should be obtained
    from a linear (ols) regression of the cloud of points of P(t)-vs-P(t-1), (1-phi)·E0 being the interception, and sigma
    being the standard deviation of the residuals of the regression.
    :param rv_params: (class containing floats) Parameters which characterize the distribution of the random variables to use
    list_profit_taking: list of real numbers which are the values to analyse for the upper threshold (profit-taking)
    :param min_num_trials: (int) Minimum number of trial calculations to do (the output includes an average and a standard deviation for the results of this number of trials).
    :param tolerance: (float) maximum difference between the results of nearby calculations to consider them converged and stop the iterations.
    :param enter_value: (float) value of the variable ("spread" in case of Ornstein-Uhlenbeck equation) to enter the portfolio.
    :return: average and standard deviation of the calculation of profits (prices or returns).
    '''

    assert (input_params.evolution_type == "Ornstein-Uhlenbeck_equation")

    strategy_duration = input_params.strategy_duration
    delt = 0.00000000000000001
    size_array_rv = 8000
    arr_profit = [] # To store the profits
    DFunit = np.exp( input_params.log_discount_factor_unit_time )
    t_in = None; p_in = None;
    ornuhl_slope = OU_params['phi']
    enter_value += OU_params['E0']    # We define the "enter_value" with respect to the mean-reverting mean
    pt += OU_params['E0'];  sl += OU_params['E0']  # Correspondingly, we must also offset the profit-taking and stop-loss parameters, because they are defined wrt to the enter_value (see function adapt_lists_thresholds).

    #print(" params OU: phi=",OU_params['phi'],"E0=",OU_params['E0'])
    #print("Enter=",enter_value, ";PT=",pt)

    poisson_probability = input_params.poisson_probability
    rv_poisson_array = None



    j=0
    for iter_ in range(min_num_trials):

        #print("NEW ITERATION\n mh=",horizon,"en=",enter_value,"pt=",pt, "sl=", sl)

        ornuhl_y0 = (1 - OU_params['phi']) * OU_params['E0']
        E0 = OU_params['E0'] # E0 is the mean the Ornstein-Uhlenbeck process tends to; it can change several times due to Poisson events (if activated). However, the investor is not aware of that changes, and hence for his decisions he uses OU_params['E0'] (wich is constant), not E0.
        if (input_params.evolution_type == "Ornstein-Uhlenbeck_equation"): p = OU_params['E0']  # We set the initial value to its expected value for simplicity
        else: p = 0
        invested = False  # This variable is true if you are owner of a position, and false otherwise (i.e. if you are not invested).
        iter_profit = 0        # Profit (evenutally cumulative) in the present iteration

        for time_ in range( strategy_duration ):

            if ((j%size_array_rv )==0): rv_array = generate_random_variable(input_params.evolution_type, rv_params, size_array_rv, j/size_array_rv )
            rv = rv_array[j%size_array_rv]

            p = ornuhl_y0 + ornuhl_slope * p + rv # p = (1-OU_params['phi'])*OU_params['E0'] + OU_params['phi']*p + rv    Price(t) given by the discretized Ornstein-Uhlenbeck equation
            #print(time_,")",p)

            # Poisson event
            if (poisson_probability != None):
                if ((j % size_array_rv) == 0): rv_poisson_array = uniform.rvs(size=size_array_rv)
                if (rv_poisson_array[j % size_array_rv] > 1 - poisson_probability):
                    if (rv_poisson_array[j % size_array_rv] > 1 - poisson_probability/2 ): # We assign 50% probability to increase or decrease of E0
                         E0 *= (1 + input_params.new_value_after_poisson_event_increase)  # xxx check!
                    else:
                         E0 *= (1 + input_params.new_value_after_poisson_event_decrease)
                    ornuhl_y0 = (1 - OU_params['phi']) * E0
                    #print("Poisson event: The E0 is now",E0 )

            j += 1

            if (not (invested)):
                if ( ((enter_value < OU_params['E0']) and (p <= enter_value+delt) and (p >= sl-delt) ) or ((enter_value >= OU_params['E0']) and (p >= enter_value-delt) and (p <= sl+delt) ) ): #and (p >= sl)
                    invested=True
                    t_in = time_
                    p_in = p
                    #print("p=",p,"Now entering")
            else: # We are invested, hence we seek to unwind our position
                if ( ( time_-t_in > horizon ) or (time_ == strategy_duration-1) or ((enter_value < OU_params['E0']) and ( (p > pt) or (p < sl) ) ) or ((enter_value >= OU_params['E0']) and ( (p < pt) or (p > sl) ) )  ):
                    DF = 1 #xxx update_discount_factor(DF, time_, DFunit, input_params.discount_factor_update_method)
                    iter_profit += (p - p_in)*DF
                    #print("Now exiting with a profit of", iter_profit, "(", p_in, ";", p, "last profit was",p-p_in,")\n")
                    invested = False
                    p_in = None
                    if (input_params.method_for_calculation_of_profits=="enter_random_one_deal"): break
                    # If stop-loss was realised, we stop investing in this cycle (iteration) beacuse the E0 may have changed
                    if (   ( (enter_value < OU_params['E0'])  and (p < sl) )  or ((enter_value >= OU_params['E0']) and (p > sl) ) ):
                        break
        if ( (input_params.evolution_type=="Ornstein-Uhlenbeck_equation") and (input_params.method_for_calculation_of_profits == "enter_random_many_deals") ):
            iter_profit *= (252/strategy_duration) # We annualize the cumulative return
        if (enter_value > OU_params['E0']-delt): iter_profit *= -1   # If we had a short position on the spread, we have to change the sign of the profit
        arr_profit.append(iter_profit)

    # Calculation of prices and Sharpe ratio for a given pair of threshold values
    results = find_measures( arr_profit )

    if (input_params.check_convergence_trading_rules):
        from pandas import DataFrame
        prod_label = OU_params['product_label']; prod_label = prod_label.replace(".csv",""); prod_label = prod_label.replace("spr_resid_","")
        filepathprofits = input_params.dir_trading_rules_convergence + "/convergence_"+prod_label+"_"+rv_params['distribution_type']+"__en_"+"{:.5f}".format(enter_value-OU_params['E0'])+"__pt_"+"{:.5f}".format(pt-OU_params['E0'])+"__sl_"+"{:.5f}".format(sl-OU_params['E0'])+".csv"
        df0 = DataFrame(arr_profit); df0.columns = ["profit_#ev_"+"{:.10f}".format(enter_value-OU_params['E0'])+"__pt_"+"{:.10f}".format(pt-OU_params['E0'])+"__sl_"+"{:.10f}".format(sl-OU_params['E0'])]
        df0.to_csv(filepathprofits,index=False)
        file_object = open(input_params.dir_trading_rules_convergence + "/list_files_to_analyse_for_convergence.csv", 'a')
        file_object.write(filepathprofits+"\n")
        file_object.close()
        del df0; del filepathprofits

    del enter_value; del pt; del sl; del input_params; del OU_params; del rv_params; del min_num_trials; del tolerance
    del arr_profit; del rv_array; del p; del time_; del p_in; del t_in; del iter_profit;

    return results

#----------------------------------------------------------------------------------------

def find_thresholds_ensured_enter_OU( mh, enter_value, pt, sl, input_params, OU_params, rv_params, min_num_trials, tolerance ):
    ''' This function finds the optimal strategy (i.e. the optimal profit-taking and stop-loss thresholds) for the given
    input parameters. It is based on the function called "batch", written by Marcos Lopez de Prado, and available in
    https://quantresearch.org/OTR.py.txt and in Chapter 13 of his book "Advances in financial machine learning".

    ===>> NOTE that this function does not take into account the frequency for entering the position. When an end (either time
    horizon, profit-taking or stop-loss) is reached, the price comes back to its starting point: << p  = enter_value >>.
    This does not evaluate how frequent {reaching that initial p} is.
    This function is very similar to Marcos Lopez de Prado's. <<======

    :param mh: Maximum horizon before unwinding (closing) the position (after entering it)
    :param enter_value: Value of the stochastic variable (spread for Ornstein-Uhlenbeck, price for "single_product") to enter a position.
    :param pt, sl: (floats) profit-taking and stop-loss thresholds to analyse.
    :param input_params: (class) User-defined input parameters for the calculation
    :param OU_params: (class containing floats) Parameters of the Ornstein-Uhlenbeck equation: "E0" is the mean-reversion parameter, "tau" is the
    half-life parameter (which determines the speed parameter phi), and "sigma" is the standard deviation of the residuals
    of the residuals of the regression of prices. The explanations on the fitting can be found in references [1] or [2].
    The discretized Ornstein-Uhlenbeck equation is: P(t) = (1-phi)·E0 + phi·P(t-1) + sigma·epsilon, where epsilon is a
    random variable with standard normal distribution. In realistic analyses the phi, E0 and sigma parameters should be obtained
    from a linear (ols) regression of the cloud of points of P(t)-vs-P(t-1), (1-phi)·E0 being the interception, and sigma
    being the standard deviation of the residuals of the regression.
    :param rv_params: (class containing floats) Parameters which characterize the distribution of the random variables to use
    list_profit_taking: list of real numbers which are the values to analyse for the upper threshold (profit-taking)
    :param min_num_trials: (int) Minimum number of trial calculations to do (the output includes an average and a standard deviation for the results of this number of trials).
    :param tolerance: (float) maximum difference between the results of nearby calculations to consider them converged and stop the iterations.
    :param enter_value: (float) initial (t=0) value of the variable ("spread" in case of Ornstein-Uhlenbeck equation).
    :return: average and standard deviation of the calculation of profits (prices or returns).
    '''

    #print("max_horizon",mh,"enter_value=",enter_value,"pt=",pt, "sl=", sl)

    assert( input_params.evolution_type=="Ornstein-Uhlenbeck_equation" )

    E0 = OU_params['E0']
    enter_value += E0   # We define the "enter_value" with respect to the mean-reverting mean
    pt += E0; sl += E0  # Correspondingly, we must also offset the profit-taking and stop-loss parameters, because they are defined wrt to the enter_value (see function adapt_lists_thresholds).

    size_array_rv = 8000
    arr_profit = []
    array_check_convergence = [-999,999]
    DFunit = np.exp( input_params.log_discount_factor_unit_time )
    ornuhl_y0 = (1 - OU_params['phi']) * OU_params['E0']
    ornuhl_slope = OU_params['phi']

    poisson_probability = input_params.poisson_probability
    value_after_poisson = input_params.new_value_after_poisson_event
    rv_poisson_array = None

    j = 0
    rv_array = generate_random_variable( input_params.evolution_type, rv_params, size_array_rv, j )

    for iter_ in range(4*min_num_trials+1):

        time_ = 0
        p  = enter_value
        DF = 1

        while True:

            j+=1
            if ((j%size_array_rv )==0): rv_array = generate_random_variable(input_params.evolution_type, rv_params, size_array_rv, j/size_array_rv )
            rv = rv_array[j%size_array_rv]

            p = ornuhl_y0 + ornuhl_slope * p + rv  # Price(t) given by the discretized Ornstein-Uhlenbeck equation: p = (1-OU_params['phi'] )*OU_params['E0'] + OU_params['phi']*p + rv

            # Poisson event
            if (poisson_probability != None):
                if ((j % size_array_rv) == 0): rv_poisson_array = uniform.rvs(size=size_array_rv)
                if (rv_poisson_array[j % size_array_rv] > 1 - poisson_probability): p = value_after_poisson

            j += 1
            time_ += 1

            #cP=(p - enter_value); if ( ( cP > pt ) or ( cP < sl ) or ( time_ > mh ) ): # MLdP original: << cP<-comb_[1] (i.e. sl) >> (negative sign)
            if (( time_ > mh ) or ((enter_value < E0) and ((p > pt) or (p < sl))) or ((enter_value >= E0) and ((p < pt) or (p > sl)))):
                DF = 1 #xxx update_discount_factor(DF, time_, DFunit, input_params.discount_factor_update_method)
                arr_profit.append( (p - enter_value)*DF )
                #print(time_,")",(p - enter_value) )
                break

        # We stop the calculation if convergence was attained. Note that the average price as a funtion of iter_ is very noisy, it may be hard to obtain convergence beyond a tolerance of E-5.
        if (iter_ == min_num_trials):
            sum_final_prices = np.sum(arr_profit)
        elif (iter_ > min_num_trials):
            sum_final_prices += (p - enter_value)*DF
            if   ( (iter_%4) ==0 ):
                array_check_convergence[ round(iter_/4) % 2] = sum_final_prices/iter_
                #print(iter_, array_check_convergence, "  ",abs(array_check_convergence[0] - array_check_convergence[1]))
                if ( abs(array_check_convergence[0]-array_check_convergence[1]) < tolerance ): break
                #xxx DEV: uncomment in the final version! if (iter_==4*min_num_trials): print("WARNING: For params "+str(pt)+", "+str(sl)+" the convergence (with tolerance "+str(tolerance)+") was not attained after",iter_,"trials. ")

    results = find_measures(arr_profit)

    del arr_profit; del rv_array; del array_check_convergence; del iter_; del p; del DF; del j; del time_
    del enter_value; del pt; del sl; del input_params; del OU_params; del rv_params; del min_num_trials; del tolerance

    return results

#----------------------------------------------------------------------------------------

def find_thresholds_single( mh, pt, sl, input_params, rv_params, min_num_trials, tolerance ):
    ''' This function finds the optimal strategy (i.e. the optimal profit-taking and stop-loss thresholds) for the given
    input parameters and the case of trading a single product. We will assume that we have a long position and that we buy
    at t=0.
    '''

    assert (input_params.evolution_type == "single_product")

    size_array_rv = 8000
    arr_profit = []
    array_check_convergence = [-999,999]
    DFunit = np.exp( input_params.log_discount_factor_unit_time )

    poisson_probability = input_params.poisson_probability
    value_after_poisson = input_params.new_value_after_poisson_event
    rv_poisson_array = None

    j = 0
    print("Params: mh",mh," pt",pt,"sl",sl)

    for iter_ in range(4*min_num_trials+1):

        time_ = 0
        p  = 1
        DF = 1

        while True:

            if ((j%size_array_rv )==0):
                rv_array   = generate_random_variable(input_params.evolution_type, rv_params, size_array_rv, j/size_array_rv )
            rv = rv_array[j%size_array_rv]

            if ( rv_params['distribution_type'] == "Gaussian"):
                p = p*(np.exp(rv))  # From closed form of geometric Brownian motion
            else:
                p * (1+rv)          # From differential equation dp = p · ( \mu · dt + \sigma · dW  )
            #print(time_,")",p)

            # Poisson event
            if (poisson_probability != None):
                if ((j % size_array_rv) == 0): rv_poisson_array = uniform.rvs(size=size_array_rv)
                if ( rv_poisson_array[j%size_array_rv] > 1- poisson_probability): p = value_after_poisson

            time_ += 1; j+=1

            if (( time_ > mh ) or (p >= pt) or (p <= sl)):
                DF = 1 #xxx update_discount_factor(DF, time_, DFunit, input_params.discount_factor_update_method)
                arr_profit.append(  p * DF - 1 ) #( p**(252/time_) ) * DF
                #print("Exiting: params:",pt,sl,"; earning: ",p-1,"\n")
                break

        # We stop the calculation if convergence was attained. Note that the average price as a funtion of iter_ is very noisy, it may be hard to obtain convergence beyond a tolerance of E-5.
        if (iter_ == min_num_trials):
            sum_final_prices = np.sum(arr_profit)
        elif (iter_ > min_num_trials):
            sum_final_prices += ( p )*DF
            if   ( (iter_%4) ==0 ):
                array_check_convergence[ round(iter_/4) % 2] = sum_final_prices/iter_
                #print(iter_, array_check_convergence, "  ",abs(array_check_convergence[0] - array_check_convergence[1]))
                if ( abs(array_check_convergence[0]-array_check_convergence[1]) < tolerance ): break
                #if (iter_==4*min_num_trials): print("WARNING: For params "+str(pt)+", "+str(sl)+" the convergence (with tolerance "+str(tolerance)+") was not attained after",iter_,"trials. ")

    results = find_measures(arr_profit)

    del arr_profit; del rv_array; del array_check_convergence; del iter_; del p; del DF; del j; del time_
    del pt; del sl; del input_params; del rv_params; del min_num_trials; del tolerance; del rv_poisson_array

    return results

#----------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------------

