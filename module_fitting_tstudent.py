'''
The functions of this module fit a given dataset ( << dataset_in >> ) to a t-student function.

Suggestion (2021-10-12): set: tolerance_fitting = 0.000005; max_n_iter = 150; n_random_tries = 150; step = uniform(0, 20).
                              starting point: [ np.median(dataset_in) ] ; [ sca_stpo, sca_stpo/8 ] ; [0] ... ; [ 2.5 ]
'''

import numpy as np
from scipy.stats import nct, cauchy
from random import uniform
from math import isnan
from module_fitting import first_iteration
from module_parameters import tolerance_fitting
np.seterr(invalid='ignore') # This tells NumPy to hide any warning with some “invalid” message in it.
np.seterr(divide='ignore')

#------------------------------------------------------------------------------------------------

class LimitsForParams:
    '''This class defines the maximum values which can be used in the fitting. Setting this aims to avoid the minimization
    to waste time analysing regions of the space parameters which are deemed to be unrealistic.'''
    def __init__(self, dataset_in, consider_skewness ):
        self.max_allowed_abs_loc = 1.5 * max( abs( np.mean( dataset_in ) ),  abs(np.median(dataset_in)) ) * 3
        ref_scale =  np.std( dataset_in[int( 0.05 * len(dataset_in) ): int( 0.95 * len(dataset_in) )] ) # Standard deviation without extreme values
        self.min_allowed_scale = ref_scale/10
        self.max_allowed_scale = ref_scale*10;
        if (consider_skewness):
            from scipy.stats import skew
            self.max_allowed_abs_skewparam = 4*abs(skew(dataset_in))
        else:
            self.max_allowed_abs_skewparam = 0;
        self.min_allowed_df = 0.5; # This is an arbitrary value
        self.max_allowed_df = 123.45678;  # This is an arbitrary value; it is necessary to impose it because the nct function fails for too high values of df

#------------------------------------------------------------------------------------------------

def find_initial_point( dataset_in, consider_skewness = False ):
    ''' This function (arbitrarily) determines the values of the starting point for the optimization algorithm.

    :param dataset_in: (numpy array of floats): The dataset whose fitting to a t-student function is sought.
           IMPORTANT: It must be sorted !!!
    :param consider_skewness: (Boolean) False if the skewness parameter to consider is zero (symmetric distribution); True otherwise.
    :return: (lists of numbers) li_locations_sg, li_scalings_sg, li_skewparams_sg, li_df_sg
    '''

    from scipy.stats import skew

    myvar = dataset_in[int(len(dataset_in) / 4): int(3 *len(dataset_in) / 4)]
    sca_stpo = 2*np.std(myvar)
    skewness_dataset = skew( dataset_in )

    li_locations_sg = [ np.median(dataset_in) ]  # [ 0, np.median(dataset_in) ]
    li_scalings_sg  = [ sca_stpo, sca_stpo/8 ]
    if ( consider_skewness == False ):
       li_skewparams_sg = [0]
    else:
       if (abs(skewness_dataset)<0.01):
           li_skewparams_sg = [0]
       else:
           if (abs(skewness_dataset)<0.1):
               li_skewparams_sg = [skewness_dataset]
           else:
               li_skewparams_sg = [ skewness_dataset, 0 ]

    li_df_sg = [2, 4, 10, 16]

    return li_locations_sg, li_scalings_sg, li_skewparams_sg, li_df_sg

#------------------------------------------------------------------------------------------------

def update_optimal_parameters( loss_fnd, loc_param_fnd, sca_param_fnd, skew_param_fnd, df_param_fnd, loss_opt, loc_param_opt, sca_param_opt, skew_param_opt, df_param_opt  ):
    '''This function updates the values of the parameters of the fitting'''
    if ( (loss_fnd < loss_opt) or (loc_param_opt == None) ):
        return loss_fnd, loc_param_fnd, sca_param_fnd, skew_param_fnd, df_param_fnd
    else:
        return loss_opt, loc_param_opt, sca_param_opt, skew_param_opt, df_param_opt

#------------------------------------------------------------------------------------------------

def fit_to_nct_global_minimum( dataset_in, n_random_tries=100, max_n_iter=150, consider_skewness=True, verbose=0 ):
    ''' This function finds the nct distribution (skewed t-student distribution) which best fits to the input dataset.
    The suffixes of variables below mean: # "_tt" means "to be tried"; "_opt" means "optimal"; "_fnd" means "found"

    :param dataset_in: (numpy array of floats): The dataset whose fitting to a t-student function is sought.
           IMPORTANT: It must be sorted !!!
    :param consider_skewness: (Boolean) False if the skewness parameter to consider is zero (symmetric distribution); True otherwise.
    :return: (4 float numbers) nct_location, nct_scaling, nct_skewparam, nct_df
    '''

    li_locations_sg, li_scalings_sg, li_skewparams_sg, li_df_sg = find_initial_point( dataset_in, consider_skewness )
    lim_params = LimitsForParams( dataset_in, consider_skewness  )
    loss_opt = 99999999; loc_param_opt = None; sca_param_opt = None; skew_param_opt = None; df_param_opt = None

    for loc_param_tt in li_locations_sg:
        for sca_param_tt in li_scalings_sg:
            for skew_param_tt in li_skewparams_sg:
                for df_param_tt in li_df_sg:
                   for trial_counter in range(n_random_tries):
                       if (verbose >= 1): print("Params IN", loc_param_tt, sca_param_tt, skew_param_tt, df_param_tt )
                       loss_fnd, loc_param_fnd, sca_param_fnd, skew_param_fnd, df_param_fnd = fit_to_nct_local_minimum( dataset_in, consider_skewness, max_n_iter, lim_params, loc_param_tt, sca_param_tt, skew_param_tt, df_param_tt, verbose )
                       loss_opt, loc_param_opt, sca_param_opt, skew_param_opt, df_param_opt = update_optimal_parameters( loss_fnd, loc_param_fnd, sca_param_fnd, skew_param_fnd, df_param_fnd, loss_opt, loc_param_opt, sca_param_opt, skew_param_opt, df_param_opt  )
                       if (verbose>=1): print("Params IN", loc_param_tt,sca_param_tt, skew_param_tt, df_param_tt, "; Params OUT",loc_param_fnd, sca_param_fnd, skew_param_fnd, df_param_fnd, "; LOSS:", loss_fnd)

    if (consider_skewness): # We repeat the calculations without skewness
        consider_skewness = False
        for loc_param_tt in li_locations_sg:
            for sca_param_tt in li_scalings_sg:
                for skew_param_tt in [0]:
                    for df_param_tt in li_df_sg:
                        for trial_counter in range( max(round(n_random_tries/2),3) ):
                            if (verbose >=1): print("Without skewness: Params IN", loc_param_tt, sca_param_tt, skew_param_tt, df_param_tt )
                            loss_fnd, loc_param_fnd, sca_param_fnd, skew_param_fnd, df_param_fnd = fit_to_nct_local_minimum(dataset_in, consider_skewness, max_n_iter, lim_params, loc_param_tt, sca_param_tt, skew_param_tt,df_param_tt, verbose)
                            loss_opt, loc_param_opt, sca_param_opt, skew_param_opt, df_param_opt = update_optimal_parameters(loss_fnd, loc_param_fnd, sca_param_fnd, skew_param_fnd, df_param_fnd, loss_opt,loc_param_opt, sca_param_opt, skew_param_opt, df_param_opt)
                            if (verbose>=1): print("Params IN", loc_param_tt,sca_param_tt, skew_param_tt, df_param_tt, "; Params OUT",loc_param_fnd, sca_param_fnd, skew_param_fnd, df_param_fnd, "; LOSS:", loss_fnd)

    print(" The GLOBAL minimum (nct) is:", "{:.9f}".format(loc_param_opt), "{:.9f}".format(sca_param_opt), "{:.9f}".format(skew_param_opt), "{:.9f}".format(df_param_opt) ,"Loss:","{:.9f}".format(loss_opt),"\n")

    del dataset_in; del n_random_tries; del max_n_iter; del consider_skewness; del verbose

    return { 'distribution_type':'nct', 'loc_param': loc_param_opt, 'scale_param':sca_param_opt, 'skewness_param':skew_param_opt, 'df_param':df_param_opt, 'loss':loss_opt }

#------------------------------------------------------------------------------------------------

def fit_to_nct_local_minimum( dataset_in, consider_skewness, max_n_iter, lim_params, loc_param0, sca_param0, skew_param0, df_param0, verbose  ):
    ''' This function finds the nct distribution (skewed t-student distribution) which best fits to the input dataset
    using a given starting point for the parameters as well as the well-known gradient descent algorithm.

    The suffixes of variables below mean: # "_tt" means "to be tried"; "_opt" means "optimal"; "_fnd" means "found"

    :param dataset_in: (numpy array of floats) set of values whose pdf is to be fit.
    :param consider_skewness: (Boolean) False if the skewness parameter to consider is zero (symmetric distribution); True otherwise.
    :return: nct_location, nct_scaling, nct_skewparam, nct_df
    '''



    # Initialization
    loss0 = 99; loss1 = 999; loss_opt = 99999999; loc_param_opt = None; sca_param_opt = None; skew_param_opt = None; df_param_opt = None;
    metropolis = 0;


    # First iteration
    loss0, loc_param0, sca_param0, skew_param0, df_param0, trash1, grad_loc0, grad_sca0, grad_skewparam0, grad_df0, trash2, loss1, loc_param1, sca_param1, skew_param1, df_param1, trash3 = first_iteration(dataset_in, 'nct', consider_skewness, loc_param0, sca_param0, skew_param0, df_param0)
    loss_opt, loc_param_opt, sca_param_opt, beta_param_opt, alpha_param_opt = update_optimal_parameters(loss1,loc_param1,sca_param1,skew_param1,df_param1,loss0,loc_param0,sca_param0,skew_param0,df_param0)

    if (None in [loc_param0, sca_param0, skew_param0, df_param0, loc_param1, sca_param1, skew_param1, df_param1]):
        del dataset_in; del max_n_iter; del consider_skewness; del lim_params;  del loc_param0; del sca_param0; del skew_param0; del df_param0; del verbose; del trash1; del trash2; del trash3
        return 9999, None, None, None, None

    # Sweeping to find the local minimum
    n_iter=0
    while ( ((( abs(loss1 - loss0) > tolerance_fitting) or (metropolis > 0.9) ) or (loss1 == 9999)  ) and (n_iter < max_n_iter) ):

        # Update the loss to check convergence
        n_iter += 1
        loss0 = loss1; metropolis = uniform(0, 1)

        # Find the gradient
        grad_loc1, grad_sca1, grad_skewparam1, grad_df1 = calculate_gradient_params( dataset_in, consider_skewness, loc_param1, sca_param1, skew_param1, df_param1 )

        # Find the step size
        step = (loc_param1 - loc_param0)   * (grad_loc1 - grad_loc0)             + (sca_param1 - sca_param0) * (grad_sca1 - grad_sca0)  + \
               (skew_param1 - skew_param0) * (grad_skewparam1 - grad_skewparam0) + (df_param1 - df_param0)   * (grad_df1 - grad_df0)
        step /= -( (grad_loc1 - grad_loc0)**2 + (grad_sca1 - grad_sca0)**2 + (grad_skewparam1 - grad_skewparam0)**2 + (grad_df1 - grad_df0)**2 )

        # Update quantities
        loc_param0 = loc_param1; sca_param0  = sca_param1;       skew_param0 = skew_param1;    df_param0 = df_param1
        grad_loc0  = grad_loc1   ; grad_sca0 = grad_sca1   ; grad_skewparam0 = grad_skewparam1; grad_df0 = grad_df1
        loc_param1  += step * grad_loc1;
        sca_param1  += step * grad_sca1;
        skew_param1 = skew_param1 * ( 1  + step * grad_skewparam1);   # 0.01*np.sign(step * grad_skewparam1)
        skew_param1 += step * grad_skewparam1;
        df_param1   += step * grad_df1
        # We comment the first line below to avoid the iterations to get stuck in this value. We keep the other three ones because they seem to improve the results in some selected test cases.
        #if (abs(loc_param1)>lim_params.max_allowed_abs_loc): loc_param1=np.sign(loc_param1)*lim_params.max_allowed_abs_loc
        sca_param1 = max( sca_param1 , lim_params.min_allowed_scale ); sca_param1 = min( sca_param1 , lim_params.max_allowed_scale );
        skew_param1 = np.sign(skew_param1) * min(abs(skew_param1), lim_params.max_allowed_abs_skewparam)
        if ( abs(df_param1-lim_params.min_allowed_df)<0.000000001 ):
             df_param1 = lim_params.min_allowed_df * (1+uniform(0,10)/100)

        # Find the loss
        try:
            loss1 = - (np.sum( np.log( nct.pdf(dataset_in, loc=loc_param1, scale=sca_param1, nc=skew_param1, df=df_param1) ) ))/len(dataset_in)
            if ( (loss1>0) or isnan(loss1) ):
                break
        except RuntimeWarning:
            break

        if (verbose>1): print(n_iter,") P. IN", loc_param0, sca_param0, skew_param0, df_param0 , "; Params OUT",  loc_param1, sca_param1, skew_param1, df_param1, "; LOSS:", loss1 )

        loss_opt, loc_param_opt, sca_param_opt, skew_param_opt, df_param_opt = update_optimal_parameters(loss1,loc_param1,sca_param1,skew_param1,df_param1,loss_opt,loc_param_opt,sca_param_opt,skew_param_opt,df_param_opt)
        #if (n_iter == max_n_iter-1): print("  WARNING: Maximum number of iterations reached.")

        df_param1 = fit_to_nct_local_minimum_sweep_df(dataset_in, max_n_iter, lim_params, loc_param1, sca_param1, skew_param1, df_param1)
        loss_opt, loc_param_opt, sca_param_opt, skew_param_opt, df_param_opt = update_optimal_parameters(loss1,loc_param1,sca_param1,skew_param1,df_param1,loss_opt,loc_param_opt,sca_param_opt,skew_param_opt,df_param_opt)

        if (consider_skewness):
            skew_param1 = fit_to_nct_local_minimum_sweep_sk(dataset_in, max_n_iter, lim_params, loc_param1, sca_param1, skew_param1, df_param1)
            loss_opt, loc_param_opt, sca_param_opt, skew_param_opt, df_param_opt = update_optimal_parameters(loss1,loc_param1,sca_param1,skew_param1,df_param1,loss_opt,loc_param_opt,sca_param_opt,skew_param_opt,df_param_opt)

    del dataset_in; del max_n_iter; del consider_skewness; del lim_params;  del loc_param0; del sca_param0; del skew_param0; del df_param0; del verbose; del trash1; del trash2; del trash3; del loss1; del loc_param1; del sca_param1; del skew_param1; del df_param1

    return loss_opt, loc_param_opt, sca_param_opt, skew_param_opt, df_param_opt

#------------------------------------------------------------------------------------------------

def calculate_gradient_params( dataset_in, consider_skewness, loc_param, sca_param, skew_param, df_param ):
    ''' This function returns the gradient of the parameters (4-component array). Note that the formula is based
    on the maximum likelihood method. The loss function is defined as minus the log of the products of all the
    values (pdf's of all points x of dataset_in), this is the sum of the logs of each pdf (-sum_i(log(pdf(x_i)))
    and hence the derivative of it wrt the 4 parameters is f'(y)/f(y), where y:=pdf(x_i).

    :param dataset_in: (numpy array of floats) set of values whose pdf is to be fit.
    :param consider_skewness: (Boolean) False if the skewness parameter to consider is zero (symmetric distribution); True otherwise.
    :param loc_param: (float) location parameter
    :param sca_param: (float) scaling parameter
    :param skew_param: (float) skewness parameter
    :param df_param: (float) number of degrees of freedom parameter
    :return: (4 float numbers) grad_loc, grad_sca, grad_skewparam, grad_df
    '''

    if (loc_param==None): loc_param=0
    if (sca_param==None): sca_param=0.005
    if (skew_param==None): skew_param=0
    if (df_param==None): df_param = 4

    h0 = sca_param  / 1000             # h0 = sca_param  / 100000
    h1 = skew_param / 20 + 1/10000000    # h1 = skew_param / 1000 + 1/10000000
    h2 = df_param   / 1000             # h2 = df_param   / 100000
    dataset_in_pdf = nct.pdf( dataset_in, loc=loc_param, scale=sca_param, nc=skew_param, df=df_param )

    grad_loc = nct.pdf( dataset_in, loc=loc_param+h0, scale=sca_param, nc=skew_param, df=df_param ) - \
               nct.pdf( dataset_in, loc=loc_param-h0, scale=sca_param, nc=skew_param, df=df_param )
    grad_loc /= (2*h0)
    grad_loc = -np.sum( grad_loc / dataset_in_pdf  )  / len(dataset_in)

    grad_sca = nct.pdf( dataset_in, loc=loc_param, scale=sca_param+h0, nc=skew_param, df=df_param ) - \
               nct.pdf( dataset_in, loc=loc_param, scale=sca_param-h0, nc=skew_param, df=df_param )
    grad_sca /= (2*h0)
    grad_sca = -np.sum( grad_sca / dataset_in_pdf  )  / len(dataset_in)

    if consider_skewness:
       grad_skewparam = nct.pdf( dataset_in, loc=loc_param, scale=sca_param, nc=skew_param+h1, df=df_param ) - \
                     nct.pdf( dataset_in, loc=loc_param, scale=sca_param, nc=skew_param-h1, df=df_param )
       grad_skewparam /= (2*h1)
       grad_skewparam = -np.sum( grad_skewparam /dataset_in_pdf  )  / len(dataset_in)
    else:
       grad_skewparam = 0

    grad_df = nct.pdf( dataset_in, loc=loc_param, scale=sca_param, nc=skew_param, df=df_param+h2 ) - \
               nct.pdf( dataset_in, loc=loc_param, scale=sca_param, nc=skew_param, df=df_param-h2 )
    grad_df /= (2*h2)
    grad_df = -np.sum( grad_df /dataset_in_pdf  )  / len(dataset_in)

    del dataset_in; del consider_skewness; del loc_param; del sca_param; del skew_param; del df_param; del h0; del h1; del h2; del dataset_in_pdf

    return grad_loc, grad_sca, grad_skewparam, grad_df

#------------------------------------------------------------------------------------------------
#
def calculate_gradient_param_sk( dataset_in, loc_param, sca_param, skew_param, df_param ):

    h1 = skew_param / 20 + 1 / 10000000
    dataset_in_pdf = nct.pdf(dataset_in, loc=loc_param, scale=sca_param, nc=skew_param, df=df_param)
    grad_skewparam = nct.pdf( dataset_in, loc=loc_param, scale=sca_param, nc=skew_param+h1, df=df_param ) - \
                 nct.pdf( dataset_in, loc=loc_param, scale=sca_param, nc=skew_param-h1, df=df_param )
    grad_skewparam /= (2*h1)
    grad_skewparam = -np.sum( grad_skewparam /dataset_in_pdf  )  / len(dataset_in)

    del dataset_in; del loc_param; del sca_param; del skew_param; del df_param

    return  grad_skewparam

#------------------------------------------------------------------------------------------------

def calculate_gradient_param_df( dataset_in, loc_param, sca_param, skew_param, df_param ):

    h2 = df_param   / 20
    dataset_in_pdf = nct.pdf( dataset_in, loc=loc_param, scale=sca_param, nc=skew_param, df=df_param )

    grad_df = nct.pdf( dataset_in, loc=loc_param, scale=sca_param, nc=skew_param, df=df_param+h2 ) - \
               nct.pdf( dataset_in, loc=loc_param, scale=sca_param, nc=skew_param, df=df_param-h2 )
    grad_df /= (2*h2)
    grad_df = -np.sum( grad_df /dataset_in_pdf  )  / len(dataset_in)

    del dataset_in; del loc_param; del sca_param; del skew_param; del df_param; del dataset_in_pdf ; del h2

    return  grad_df

#------------------------------------------------------------------------------------------------

def fit_to_nct_local_minimum_sweep_sk( dataset_in, max_n_iter, lim_params, loc_param0, sca_param0, skew_param0, df_param0  ):
    ''' This function finds the nct distribution (skewed t-student distribution) which best fits to the input dataset
    using a given starting point for the parameters as well as the well-known gradient descent algorithm.

    The suffixes of variables below mean: # "_tt" means "to be tried"; "_opt" means "optimal"; "_fnd" means "found"

    :param dataset_in: (numpy array of floats) set of values whose pdf is to be fit.
    :param consider_skewness: (Boolean) False if the skewness parameter to consider is zero (symmetric distribution); True otherwise.
    :return: nct_location, nct_scaling, nct_skewparam, nct_df
    '''

    # Initialization
    loss0 = 99; loss1 = 999; loss_opt = 99999999; loc_param_opt = None; sca_param_opt = None; skew_param_opt = None; df_param_opt = None;
    metropolis = 0;
    skew_param00 = skew_param0


    # First iteration
    grad_skewparam0 = calculate_gradient_param_sk(dataset_in, loc_param0, sca_param0, skew_param0, df_param0 )
    step = -1

    if (abs(step)>200): step = 200*np.sign(step)
    loc_param1  = loc_param0
    sca_param1  = sca_param0
    skew_param1 = skew_param0 + step * grad_skewparam0
    df_param1   = df_param0
    skew_param1 = np.sign(skew_param1) * min( abs( skew_param1 ), lim_params.max_allowed_abs_skewparam )

    # Sweeping to find the local minimum
    n_iter=0
    while ( ((( abs(loss1 - loss0) > tolerance_fitting) or (metropolis > 0.9) ) or (loss1 == 9999)  ) and (n_iter < max_n_iter) ):

        # Update the loss to check convergence
        n_iter += 1
        loss0 = loss1; metropolis = uniform(0, 1)

        # Find the gradient
        grad_skewparam1  = calculate_gradient_param_sk( dataset_in, loc_param1, sca_param1, skew_param1, df_param1 )

        # Find the step size
        step = - (skew_param1 - skew_param0) * (grad_skewparam1 - grad_skewparam0) / (  (grad_skewparam1 - grad_skewparam0)**2  )

        # Update quantities
        skew_param0 = skew_param1; grad_skewparam0 = grad_skewparam1
        skew_param1 = skew_param1 * ( 1 + step * grad_skewparam1);   skew_param1 += step * grad_skewparam1;
        skew_param1 = np.sign(skew_param1) * min(abs(skew_param1), lim_params.max_allowed_abs_skewparam)

        # Find the loss
        try:
            loss1 = - (np.sum( np.log( nct.pdf(dataset_in, loc=loc_param1, scale=sca_param1, nc=skew_param1, df=df_param1) ) ))/len(dataset_in)
            if (loss1>0):break
        except RuntimeWarning:
            break
        if (isnan(loss1)):
            break

        loss_opt, loc_param_opt, sca_param_opt, skew_param_opt, df_param_opt = update_optimal_parameters(loss1,loc_param1,sca_param1,skew_param1,df_param1,loss_opt,loc_param_opt,sca_param_opt,skew_param_opt,df_param_opt)

    if ((skew_param_opt == None) or (isnan(skew_param_opt))): skew_param_opt = skew_param00

    del dataset_in; del max_n_iter; del lim_params; del loc_param0; del sca_param0; del skew_param0;del skew_param00; del df_param0

    return  skew_param_opt

#------------------------------------------------------------------------------------------------

def fit_to_nct_local_minimum_sweep_df( dataset_in, max_n_iter, lim_params, loc_param0, sca_param0, skew_param0, df_param0  ):

    # Initialization
    loss0 = 99; loss1 = 999; loss_opt = 99999999; loc_param_opt = None; sca_param_opt = None; skew_param_opt = None; df_param_opt = None;
    metropolis = 0;

    # First iteration
    grad_df0 = calculate_gradient_param_df( dataset_in, loc_param0, sca_param0, skew_param0, df_param0 )
    step = -1

    if (abs(step)>200): step = 200*np.sign(step)
    loc_param1  = loc_param0
    sca_param1  = sca_param0
    skew_param1 = skew_param0
    df_param1   = df_param0  + step * grad_df0
    df_param1 = max(df_param1, lim_params.min_allowed_df)
    df_param1 = min(df_param1, lim_params.max_allowed_df)

    # Sweeping to find the local minimum
    n_iter=0
    while ( ((( abs(loss1 - loss0) > tolerance_fitting) or (metropolis > 0.9) ) or (loss1 == 9999)  ) and (n_iter < max_n_iter) ):

        # Update the loss to check convergence
        n_iter += 1
        loss0 = loss1; metropolis = uniform(0, 1)

        # Find the gradient
        grad_df1 = calculate_gradient_param_df( dataset_in, loc_param1, sca_param1, skew_param1, df_param1 )

        # Find the step size
        step =  -(df_param1 - df_param0)   * (grad_df1 - grad_df0) / (  (grad_df1 - grad_df0)**2 )

        # Update quantities
        df_param1   += step * grad_df1
        df_param1 = max(df_param1, lim_params.min_allowed_df)
        df_param1 = min(df_param1, lim_params.max_allowed_df)

        # Find the loss
        try:
            loss1 = - (np.sum( np.log( nct.pdf(dataset_in, loc=loc_param1, scale=sca_param1, nc=skew_param1, df=df_param1) ) ))/len(dataset_in)
        except RuntimeWarning:
            break
        if (isnan(loss1)): break

        loss_opt, loc_param_opt, sca_param_opt, skew_param_opt, df_param_opt = update_optimal_parameters(loss1,loc_param1,sca_param1,skew_param1,df_param1,loss_opt,loc_param_opt,sca_param_opt,skew_param_opt,df_param_opt)

    if  ( (df_param_opt==None) or (isnan(df_param_opt) ) ) : df_param_opt = df_param0

    del dataset_in; del max_n_iter; del lim_params; del loc_param0; del sca_param0; del skew_param0; del df_param0

    return  df_param_opt

#------------------------------------------------------------------------------------------------
