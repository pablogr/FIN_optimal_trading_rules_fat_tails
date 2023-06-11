'''
The functions of this module fit a given dataset ( << dataset_in >> ) to a t-student function.

Suggestion (2021-10-12): set: tolerance_fitting = 0.000005; max_n_iter = 150; n_random_tries = 150; step = uniform(0, 20).
                              starting guess: [ np.median(dataset_in) ] ; [ sca_guess, sca_guess/8 ] ; [0] ... ; [ 2.5 ]
'''


from scipy.stats import genhyperbolic, cauchy
import numpy as np
from numpy.random import uniform
from math import isnan
np.seterr(invalid='ignore') # This tells NumPy to hide any warning with some “invalid” message in it.
np.seterr(divide='ignore')
from random import uniform
from module_fitting import first_iteration
from module_parameters import tolerance_fitting

#np.random.seed(seed=1234366); random.seed(1033)

#------------------------------------------------------------------------------------------------

class LimitsForParams:
    '''This class defines the maximum values which can be used in the fitting. Setting this aims to avoid the minimization
    to waste time analysing regions of the space parameters which are deemed to be unrealistic.'''
    def __init__(self, dataset_in, consider_skewness ):
        self.max_allowed_abs_loc = 1.5 * max( abs( np.mean( dataset_in ) ),  abs(np.median(dataset_in)) )
        ref_scale =  np.std( dataset_in[int( 0.05 * len(dataset_in) ): int( 0.95 * len(dataset_in) )] ) # Standard deviation without extreme values
        self.min_allowed_scale = ref_scale/10
        self.max_allowed_scale = ref_scale*10;
        if (consider_skewness):
            from scipy.stats import skew
            self.skew_dataset_in = skew(dataset_in)
            self.max_allowed_abs_skewparam = 2*abs(self.skew_dataset_in)
        else:
            self.max_allowed_abs_skewparam = 0;
        self.min_allowed_abs_a = 0.000000001;   # This is an arbitrary value
        self.max_allowed_abs_a = 100;  # This is an arbitrary value
        #self.min_allowed_abs_p = 0.000000001;   # This is an arbitrary value
        #self.max_allowed_abs_p = 1000;           # This is an arbitrary value
        del dataset_in; del consider_skewness

#------------------------------------------------------------------------------------------------

def find_starting_point( dataset_in, consider_skewness = True, consider_nonzero_p=True ):
    ''' This function (arbitrarily) determines the values of the starting point for the optimization algorithm.

    :param dataset_in: (numpy array of floats): The dataset whose fitting to a t-student function is sought.
           IMPORTANT: It must be sorted !!!
    :param consider_skewness: (Boolean) False if the b parameter to consider is zero (symmetric distribution); True otherwise.
    :return: (lists of numbers) li_locations_sg, li_scalings_sg, li_b_params_sg, li_a_params_sg, li_p_params_sg
    '''

    from scipy.stats import skew

    myvar = dataset_in[int(len(dataset_in) / 4): int(3 *len(dataset_in) / 4)]
    sca_guess = 2*np.std(myvar)
    skewness_dataset = skew( dataset_in )

    li_locations_sg = [ np.median(dataset_in) ]   
    li_scalings_sg  = [ sca_guess ]
    if ( consider_skewness == False ):
       li_b_params_sg = [0]
    else:
       if (abs(skewness_dataset)<0.01):
           li_b_params_sg = [0]
       else:
           if (abs(skewness_dataset)<0.1):
               li_b_params_sg = [ skewness_dataset, 0 ]
           else:
               li_b_params_sg = [ skewness_dataset, skewness_dataset/2, 0, -skewness_dataset ]

    li_a_params_sg = [ 0.02, 0.6 ]

    if (consider_nonzero_p):
        li_p_params_sg = [ -2.5, -1.5, 0.4 ]
    else:
        li_p_params_sg = [ 0 ]

    del dataset_in; del consider_skewness; del myvar; del skewness_dataset

    return li_locations_sg, li_scalings_sg, li_b_params_sg, li_a_params_sg, li_p_params_sg

#------------------------------------------------------------------------------------------------

def update_optimal_parameters( loss_fnd, loc_param_fnd, sca_param_fnd, b_param_fnd, a_param_fnd, p_param_fnd, loss_opt, loc_param_opt, sca_param_opt, b_param_opt, a_param_opt, p_param_opt  ):
    '''This function updates the values of the parameters of the fitting'''
    if ( (loss_fnd < loss_opt) or (loc_param_opt == None) ):
        del loss_opt; del loc_param_opt; del sca_param_opt; del b_param_opt; del a_param_opt; del p_param_opt
        return loss_fnd, loc_param_fnd, sca_param_fnd, b_param_fnd, a_param_fnd, p_param_fnd
    else:
        del loss_fnd; del loc_param_fnd; del sca_param_fnd; del b_param_fnd; del a_param_fnd; del p_param_fnd
        return loss_opt, loc_param_opt, sca_param_opt, b_param_opt, a_param_opt, p_param_opt

#------------------------------------------------------------------------------------------------

def fit_to_genhyperbolic_global_minimum( dataset_in,  n_random_tries,  max_n_iter, consider_skewness=True, consider_nonzero_p=True, verbose=0 ):
    ''' This function finds the generalized hyperbolic distribution which best fits to the input dataset.
    The suffixes of variables below mean: # "_tt" means "to be tried"; "_opt" means "optimal"; "_fnd" means "found"

    :param dataset_in: (numpy array of floats): The dataset whose fitting to a t-student function is sought.
                       IMPORTANT: It must be sorted !!!
    :param consider_skewness: (Boolean) False if the skewness parameter to consider is zero (symmetric distribution); True otherwise.
    :param consider_nonzero_p: (Boolean) False if the p parameter to consider is zero; True otherwise.
    :return: (4 float numbers) nct_location, nct_scaling, nct_skewparam, nct_df
    '''

    #xxx
    n_random_tries = 6; max_n_iter = 16

    li_locations_sg, li_scalings_sg, li_b_sg, li_a_sg, li_p_sg = find_starting_point( dataset_in, consider_skewness, consider_nonzero_p )
    lim_params = LimitsForParams(dataset_in, consider_skewness)

    loss_opt = 99999999; loc_param_opt = None; sca_param_opt = None; b_param_opt = None; a_param_opt = None; p_param_opt = None

    for loc_param_tt in li_locations_sg:
        for sca_param_tt in li_scalings_sg:
            for b_param_tt in li_b_sg:
                for a_param_tt in li_a_sg:
                  for p_param_tt in li_p_sg:
                      if (verbose>1): print("Params IN", loc_param_tt, sca_param_tt, b_param_tt, a_param_tt, p_param_tt)
                      for trial_counter in range(n_random_tries):

                          loss_fnd, loc_param_fnd, sca_param_fnd, b_param_fnd, a_param_fnd, p_param_fnd = fit_to_genhyperbolic_local_minimum( dataset_in, max_n_iter, consider_skewness, lim_params, consider_nonzero_p, loc_param_tt, sca_param_tt, b_param_tt, a_param_tt , p_param_tt, verbose )
                          loss_opt, loc_param_opt, sca_param_opt, b_param_opt, a_param_opt, p_param_opt  = update_optimal_parameters( loss_fnd, loc_param_fnd, sca_param_fnd, b_param_fnd, a_param_fnd, p_param_fnd,
                                                                                                                                      loss_opt, loc_param_opt, sca_param_opt, b_param_opt, a_param_opt, p_param_opt  )
                          if (verbose>0): print("Params IN",  loc_param_tt, sca_param_tt, b_param_tt, a_param_tt , p_param_tt , "; Params OUT", loc_param_fnd, sca_param_fnd, b_param_fnd, a_param_fnd, p_param_fnd, "; LOSS:", loss_fnd )

    print(" The GLOBAL minimum (genh) is:", loc_param_opt,  sca_param_opt, b_param_opt,  a_param_opt, p_param_opt,"; Loss:", loss_opt,"\n" )

    del dataset_in; del n_random_tries; del  max_n_iter; del consider_skewness; del consider_nonzero_p; del verbose

    return { 'distribution_type':'genhyperbolic', 'loc_param': loc_param_opt, 'scale_param':sca_param_opt, 'b_param':b_param_opt, 'a_param':a_param_opt,'p_param':p_param_opt, 'loss':loss_opt }

#------------------------------------------------------------------------------------------------

def capfloor_params(lim_params,  loc_param1, sca_param1, b_param1, a_param1, p_param1 ):

    if (None in [loc_param1, sca_param1, b_param1, a_param1, p_param1]):
        return loc_param1, sca_param1, b_param1, a_param1, p_param1

    loc_param1 = np.sign(loc_param1) * min(abs(loc_param1), lim_params.max_allowed_abs_loc)
    sca_param1 = max(sca_param1, lim_params.min_allowed_scale); sca_param1 = min(sca_param1, lim_params.max_allowed_scale);
    b_param1 = np.sign(b_param1) * min( abs( b_param1 ), lim_params.max_allowed_abs_skewparam )
    a_param1 = np.sign(a_param1) * min( abs(a_param1), lim_params.max_allowed_abs_a)
    a_param1 = np.sign(a_param1) * max( abs(a_param1), lim_params.min_allowed_abs_a)
    #p_param1 = np.sign(p_param1) * min( abs(p_param1), lim_params.max_allowed_abs_p)
    #p_param1 = np.sign(p_param1) * max( abs(p_param1), lim_params.min_allowed_abs_p)

    # The generalized hyperbolic distribution demands that |a|>=|b|, otherwise complex numbers appear. We modify a because b varies slowly.
    if ( (a_param1!= None) and (b_param1!= None) and ( abs(b_param1) >= abs(a_param1) ) ):
        a_param1 *= uniform(1.1,4) * abs(b_param1)/ ( abs(a_param1) )

    del lim_params

    return loc_param1, sca_param1, b_param1, a_param1, p_param1

#------------------------------------------------------------------------------------------------

def fit_to_genhyperbolic_local_minimum( dataset_in, max_n_iter, consider_skewness, lim_params, consider_nonzero_p, loc_param0, sca_param0, b_param0, a_param0, p_param0, verbose=0  ):
    ''' This function finds the generalized hyperbolic distribution which best fits to the input dataset.

    The suffixes of variables below mean: # "_tt" means "to be tried"; "_opt" means "optimal"; "_fnd" means "found"

    :param dataset_in: (numpy array of floats) set of values whose pdf is to be fit.
    :param consider_skewness: (Boolean) False if the skewness parameter to consider is zero (symmetric distribution); True otherwise.
    :param consider_nonzero_p: (Boolean) False if the p parameter to consider is zero; True otherwise.
    :return: loss_opt, loc_param_opt, sca_param_opt, b_param_opt, a_param_opt, p_param_opt
    '''

    #np.random.seed(seed=1234366); random.seed(1033)

    # Initialization
    loss0 = 99; loss1 = 999; loss_opt = 99999999; loc_param_opt = None; sca_param_opt = None; b_param_opt = None; a_param_opt = None; p_param_opt = None;
    metropolis = 0;
    list_single_sweep = ['a'];
    if (consider_skewness): list_single_sweep.append('b')
    if ( consider_nonzero_p ): list_single_sweep.append('p')

    # First iteration
    loss0, loc_param0, sca_param0, b_param0, a_param0, p_param0, grad_loc0, grad_sca0, grad_b0, grad_a0, grad_p0, loss1, loc_param1, sca_param1, b_param1, a_param1, p_param1 = first_iteration(dataset_in, 'genhyperbolic', consider_skewness, loc_param0, sca_param0, b_param0, a_param0, p_param0, consider_nonzero_p )

    #if (None in [loss0, loc_param0, sca_param0, b_param0, a_param0, p_param0, grad_loc0, grad_sca0, grad_b0, grad_a0, grad_p0, loss1, loc_param1, sca_param1, b_param1, a_param1, p_param1]):
    #    return 9999999, None, None, None, None, None

    loss_opt, loc_param_opt, sca_param_opt, b_param_opt, a_param_opt, p_param_opt = update_optimal_parameters(loss1,loc_param1,sca_param1,b_param1,a_param1,p_param1,loss0,loc_param0,sca_param0,b_param0,a_param0,p_param0)

    loc_param1, sca_param1, b_param1, a_param1, p_param1 = capfloor_params(lim_params, loc_param1, sca_param1, b_param1, a_param1, p_param1)

    if ((isnan(loss0)) or (None in [loc_param0, sca_param0, b_param0, a_param0, p_param0, loc_param1, sca_param1, b_param1, a_param1, p_param1]) ):
        print("\nWARNING: Could not find a fitting to genhyperbolic for input parameters ", loc_param0, sca_param0, b_param0, a_param0, p_param0)
        del dataset_in; del max_n_iter; del consider_skewness; del lim_params; del consider_nonzero_p; del loc_param0; del sca_param0; del b_param0; del a_param0; del p_param0; del verbose
        return 9999, None, None, None, None, None


    # Sweeping to find the local minimum
    n_iter=0
    while ( ( (( abs(loss1 - loss0) > tolerance_fitting) or (metropolis > 0.9) ) or (loss1 == 9999) ) and (n_iter < max_n_iter) ):

        # Update the loss to check convergence
        n_iter += 1
        loss0 = loss1; metropolis = uniform(0, 1)

        # Find the gradient
        grad_loc1, grad_sca1, grad_b1, grad_a1 , grad_p1 = calculate_gradient_params( dataset_in, consider_skewness, consider_nonzero_p, loc_param1, sca_param1, b_param1, a_param1, p_param1 )
        if (None in [grad_loc1, grad_sca1, grad_b1, grad_a1 , grad_p1]):
            return 9999999, None, None, None, None, None

        # Find the step size
        step = (loc_param1 - loc_param0) * (grad_loc1 - grad_loc0) + (sca_param1 - sca_param0) * (grad_sca1 - grad_sca0)  + \
               (b_param1 - b_param0) * (grad_b1 - grad_b0) + (a_param1 - a_param0) * (grad_a1 - grad_a0) + (p_param1 - p_param0) * (grad_p1 - grad_p0)
        step /= -( (grad_loc1 - grad_loc0)**2 + (grad_sca1 - grad_sca0)**2 + (grad_b1 - grad_b0)**2 + (grad_a1 - grad_a0)**2 + (grad_p1 - grad_p0)**2 )

        # Update quantities
        loc_param0 = loc_param1;   sca_param0 = sca_param1;  b_param0 = b_param1;  a_param0 = a_param1;  p_param0 = p_param1
        grad_loc0  = grad_loc1   ; grad_sca0  = grad_sca1   ; grad_b0 = grad_b1;    grad_a0 = grad_a1;    grad_p0 = grad_p1
        loc_param1  += step * grad_loc1;  sca_param1  += step * grad_sca1;  b_param1 += step * grad_b1;  a_param1 += step * grad_a1;  p_param1 += step * grad_p1
        loc_param1, sca_param1, b_param1, a_param1, p_param1 = capfloor_params(lim_params, loc_param1, sca_param1, b_param1, a_param1, p_param1)

        # Find the loss
        try:
            loss1 = - (np.sum( np.log( genhyperbolic.pdf(dataset_in, loc=loc_param1, scale=sca_param1, b=b_param1, a=a_param1, p=p_param1 ) ) ))/len(dataset_in)
            if ((loss1 > 0) or isnan(loss1)):
                break
        except RuntimeWarning:
            break #loc_param1 = uniform(-1,1); sca_param1 = uniform(0.2,2); b_param1 = 0 ; a_param1 = uniform(2,4); p_param1 = 0

        if (verbose > 0): print( "iter ",n_iter,") Params",loc_param1, sca_param1,b_param1,a_param1,p_param1,"; loss=",loss1)
        loss_opt, loc_param_opt, sca_param_opt, b_param_opt, a_param_opt, p_param_opt = update_optimal_parameters(loss1,loc_param1,sca_param1,b_param1,a_param1,p_param1,loss_opt,loc_param_opt,sca_param_opt,b_param_opt,a_param_opt,p_param_opt)

        #loss_opt, loc_param_opt, sca_param_opt, b_param_opt, a_param_opt, p_param_opt = update_optimal_parameters(loss1,loc_param1,sca_param1,b_param1,a_param1,p_param1,loss_opt,loc_param_opt,sca_param_opt,b_param_opt,a_param_opt,p_param_opt)

        #a_param1 = fit_to_genhyperbolic_local_minimum_sweep_a(dataset_in, max_n_iter, lim_params, loc_param1, sca_param1, b_param1, a_param1, p_param1 )
        for param_single_sweep in list_single_sweep:
            if ((param_single_sweep=='b') and ((n_iter%3)==0)):
                loss1, b_param1 = fit_to_genhyperbolic_local_minimum_sweep_b(dataset_in, lim_params, loc_param1, sca_param1, b_param1, a_param1, p_param1, loss1)
            else:
                loss1,loc_param1,sca_paramx,b_param1,a_param1,p_param1 = fit_to_genhyperbolic_local_minimum_sweep_single_param(param_single_sweep, dataset_in, max_n_iter, consider_skewness, consider_nonzero_p, lim_params, loc_param1,sca_param1,b_param1,a_param1,p_param1  )
            loss_opt, loc_param_opt, sca_param_opt, b_param_opt, a_param_opt, p_param_opt = update_optimal_parameters(loss1,loc_param1,sca_param1,b_param1,a_param1,p_param1,loss_opt,loc_param_opt,sca_param_opt, b_param_opt,a_param_opt,p_param_opt)

    del dataset_in; del max_n_iter; del consider_skewness; del lim_params; del consider_nonzero_p; del loc_param0; del sca_param0; del b_param0; del a_param0; del p_param0

    return loss_opt, loc_param_opt, sca_param_opt, b_param_opt, a_param_opt, p_param_opt

#----------------------------------------------------------------------------------------------------------------------

def fit_to_genhyperbolic_local_minimum_sweep_single_param( sp, dataset_in, max_n_iter, consider_skewness, consider_nonzero_p, lim_params, loc_param0, sca_param0, b_param0, a_param0, p_param0  ):
    '''
    '''

    #np.random.seed(seed=1234366);random.seed(1033)

    from random import uniform
    from math import isnan


    # Initialization
    loss0 = 99; loss1 = 999;  loc_param_opt = loc_param0; sca_param_opt = sca_param0; b_param_opt = b_param0; a_param_opt = a_param0; p_param_opt = p_param0;
    loss_opt = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param0, scale=sca_param0, b=b_param0, a=a_param0,p=p_param0)))) / len(dataset_in)
    metropolis = 0;

    # First iteration
    grad_sp0 = calculate_gradient_single_param( sp, dataset_in, consider_skewness, consider_nonzero_p, lim_params, loc_param0, sca_param0, b_param0, a_param0, p_param0 )
    step =  uniform(0.0001, 1)*np.sign(uniform(-1,1))
    updp0 = {'a': a_param0, 'b': b_param0, 'p': p_param0} # Updated parameters
    updp1 = updp0.copy()
    updp1[sp] += step * grad_sp0
    loc_param0, sca_param0, updp1['b'], updp1['a'], updp1['p'] = capfloor_params(lim_params, loc_param0, sca_param0, updp1['b'],updp1['a'], updp1['p'])

    # Sweeping to find the local minimum
    n_iter=0
    while ( ( (( abs(loss1 - loss0) > tolerance_fitting) or (metropolis > 0.9) ) or (loss1 == 9999) ) and (n_iter < max_n_iter) ):

        n_iter += 1
        loss0 = loss1; metropolis = uniform(0, 1)

        grad_sp1 = calculate_gradient_single_param(sp, dataset_in, consider_skewness, consider_nonzero_p, lim_params, loc_param0, sca_param0, updp1['b'] , updp1['a'], updp1['p'] )
        step = - (updp1['b']-updp0['b']+updp1['a']-updp0['a']+updp1['p']-updp0['p']) * (grad_sp1 - grad_sp0) / ( (grad_sp1 - grad_sp0)**2 )
        updp0    = updp1.copy()
        grad_sp0 = grad_sp1
        updp1[sp] += step * grad_sp1

        loc_param0, sca_param0, updp1['b'], updp1['a'], updp1['p'] = capfloor_params(lim_params, loc_param0, sca_param0, updp1['b'],updp1['a'], updp1['p'])

        # Find the loss
        try:
            loss1 = - (np.sum( np.log( genhyperbolic.pdf(dataset_in, loc=loc_param0, scale=sca_param0, b=updp1['b'], a=updp1['a'], p=updp1['p'] ) ) ))/len(dataset_in)
            #print("  --- En el try: Loss",loss1,"Params: loc=",loc_param0, "scale",sca_param0,"b",updp1['b'], "a",updp1['a'], "p",updp1['p'])
            if ((loss1 > 0) or isnan(loss1)):
                break
        except RuntimeWarning:
            break #loc_param1 = uniform(-1,1); sca_param1 = uniform(0.2,2); b_param1 = 0 ; a_param1 = uniform(2,4); p_param1 = 0

        loss_opt, loc_param_opt, sca_param_opt, b_param_opt, a_param_opt, p_param_opt = update_optimal_parameters(loss1,loc_param0,sca_param0,updp1['b'],updp1['a'],updp1['p'],loss_opt,loc_param_opt,sca_param_opt,b_param_opt,a_param_opt,p_param_opt)
        # if (n_iter == max_n_iter-1): print("  WARNING: Maximum number of iterations reached.")

    del dataset_in; del max_n_iter; del lim_params;  del loc_param0; del sca_param0; del b_param0; del a_param0; del p_param0

    return  loss_opt, loc_param_opt, sca_param_opt, b_param_opt, a_param_opt, p_param_opt

#------------------------------------------------------------------------------------------------

def fit_to_genhyperbolic_local_minimum_sweep_b( dataset_in, lim_params, loc_param0, sca_param0, b_param0, a_param0, p_param0, loss0  ):
    ''' This function sweeps the values of the skewness parameter using a equipaced grid. This is found to be more effective
    (for the b parameter) than either doing gradient descent in all 5 parameters or in the b parameter alone (in both latter
    cases the b parameter hardly changes).
    Example:
        iter  1 ) Params 0.0013263767625094906 0.011543424896024449 0.1593836498447446 0.6822689618860058 -0.008124297673654872 ; loss= -2.4817292105199233
        Optimal prev = 0.1593836498447446 Loss -2.4817292105199233
        Optimal found= -0.04722265141522439 Loss -2.53304951629618
    '''

    loss_opt = loss0; loc_param_opt=loc_param0; sca_param_opt=sca_param0;  b_param_opt=b_param0;a_param_opt=a_param0; p_param_opt=p_param0
    if (abs(lim_params.skew_dataset_in))>0.1:   n_essais = 50
    elif (abs(lim_params.skew_dataset_in))>0.08: n_essais = 40
    elif (abs(lim_params.skew_dataset_in)) > 0.05: n_essais = 30
    else: n_essais = 20

    for i in range(n_essais):
        my_b = lim_params.skew_dataset_in * (i- int(n_essais/4)) / int(n_essais/2)
        loss1 = - (np.sum(np.log(genhyperbolic.pdf(dataset_in, loc=loc_param0, scale=sca_param0, b=my_b, a=a_param0,p=p_param0)))) / len(dataset_in)
        loss_opt, loc_param_opt, sca_param_opt, b_param_opt, a_param_opt, p_param_opt = update_optimal_parameters(loss1,loc_param0,sca_param0,my_b, a_param0,p_param0,loss_opt, loc_param_opt,sca_param_opt,b_param_opt,a_param_opt,p_param_opt)

    del dataset_in; del lim_params;  del loc_param0; del sca_param0;  del a_param0; del p_param0; del my_b; del n_essais

    return  loss_opt, b_param_opt

#------------------------------------------------------------------------------------------------

def calculate_gradient_params( dataset_in, consider_skewness, consider_nonzero_p, loc_param, sca_param, b_param, a_param, p_param ):
    ''' This function retunrs the gradient of the parameters (4-component array). Note that the formula is based
    on the maximum likelihood method. The loss function is defined as minus the log of the products of all the
    values (pdf's of all points x of datasetin), this is the sum of the logs of each pdf (-sum_i(log(pdf(x_i)))
    and hence the derivative of it wrt the 4 parameters is f'(y)/f(y), where y:=pdf(x_i).

    :param dataset_in: (numpy array of floats) set of values whose pdf is to be fit.
    :param consider_skewness: (Boolean) False if the skewness parameter to consider is zero (symmetric distribution); True otherwise.
    :param consider_nonzero_p: (Boolean) False if the p parameter to consider is zero; True otherwise.
    :param loc_param: (float) location parameter
    :param sca_param: (float) scaling parameter
    :param b_param: (float) skewness parameter
    :param a_param: (float) a parameter
    :param p_param: (float) p parameter
    :return: (5 float numbers) grad_loc, grad_sca, grad_b, grad_a, grad_p
    '''

    if (loc_param == None): loc_param = 0
    if (sca_param==None):   sca_param = (2*np.std(dataset_in[int(len(dataset_in) / 4): int(3 *len(dataset_in) / 4)]))*uniform(0.5,1.25)
    if (a_param == None):   a_param = uniform(1.05,1.95) #1.6
    if (b_param==None):     b_param=0
    if (p_param==None):     p_param=0

    h0 = sca_param  / 100000
    h1 = b_param / 100 + 1/1000
    h2 = a_param   / 100000
    dataset_in_pdf = genhyperbolic.pdf( dataset_in, loc=loc_param, scale=sca_param, b=b_param, a=a_param , p=p_param )

    grad_loc = genhyperbolic.pdf( dataset_in, loc=loc_param+h0, scale=sca_param, b=b_param, a=a_param, p=p_param ) - \
               genhyperbolic.pdf( dataset_in, loc=loc_param-h0, scale=sca_param, b=b_param, a=a_param, p=p_param )
    grad_loc /= (2*h0)
    grad_loc = -np.sum( grad_loc / dataset_in_pdf  )  / len(dataset_in)

    grad_sca = genhyperbolic.pdf( dataset_in, loc=loc_param, scale=sca_param+h0, b=b_param, a=a_param, p=p_param ) - \
               genhyperbolic.pdf( dataset_in, loc=loc_param, scale=sca_param-h0, b=b_param, a=a_param, p=p_param )
    grad_sca /= (2*h0)
    grad_sca = -np.sum( grad_sca / dataset_in_pdf  )  / len(dataset_in)

    if consider_skewness:
       grad_b = genhyperbolic.pdf( dataset_in, loc=loc_param, scale=sca_param, b=b_param+h1, a=a_param, p=p_param ) - \
                genhyperbolic.pdf( dataset_in, loc=loc_param, scale=sca_param, b=b_param-h1, a=a_param, p=p_param )
       grad_b /= (2*h1)
       grad_b = -np.sum( grad_b /dataset_in_pdf  )  / len(dataset_in)
    else:
       grad_b = 0

    grad_a =   genhyperbolic.pdf( dataset_in, loc=loc_param, scale=sca_param, b=b_param, a=a_param+h2, p=p_param ) - \
               genhyperbolic.pdf( dataset_in, loc=loc_param, scale=sca_param, b=b_param, a=a_param-h2, p=p_param  )
    grad_a /= (2*h2)
    grad_a = -np.sum( grad_a /dataset_in_pdf  )  / len(dataset_in)

    if consider_nonzero_p:
       grad_p =   genhyperbolic.pdf( dataset_in, loc=loc_param, scale=sca_param, b=b_param, a=a_param, p=p_param+h2 ) - \
                  genhyperbolic.pdf( dataset_in, loc=loc_param, scale=sca_param, b=b_param, a=a_param, p=p_param-h2  )
       grad_p /= (2*h2)
       grad_p = -np.sum( grad_p /dataset_in_pdf  )  / len(dataset_in)
    else:
       grad_p = 0

    del dataset_in; del consider_skewness; del consider_nonzero_p; del loc_param; del sca_param; del b_param; del a_param; del p_param; del h0; del h1; del h2

    return grad_loc, grad_sca, grad_b, grad_a, grad_p

#------------------------------------------------------------------------------------------------

def calculate_gradient_single_param( sp, dataset_in, consider_skewness, consider_nonzero_p, lim_params, loc_param, sca_param, b_param, a_param, p_param ):
    '''   sp (single param, whose gradient is calculated) must be either 'a', 'b' or 'p'.
    '''

    if ( (not consider_skewness) and (sp=='b') ): return 0
    if ( (not consider_nonzero_p) and (sp=='p')): return 0


    if   (sp== 'a'):
        h = a_param   / 200 + 1/1000
        updp_p = { 'a': a_param+h, 'b':b_param , 'p':p_param} # Updated parameters (plus and minus)
        updp_m = { 'a': a_param-h, 'b':b_param,  'p':p_param}
    elif (sp== 'b'):
        h = b_param / 200 + 1/10000
        updp_p = {'a': a_param, 'b': b_param + h, 'p': p_param}
        updp_m = {'a': a_param, 'b': b_param - h, 'p': p_param}
    elif (sp== 'p'):
        h = p_param / 200 + 1/1000
        updp_p = {'a': a_param, 'b': b_param, 'p': p_param + h}
        updp_m = {'a': a_param, 'b': b_param, 'p': p_param - h}

    loc_param0, sca_param0, updp_p['b'], updp_p['a'], updp_p['p'] = capfloor_params(lim_params, loc_param, sca_param, updp_p['b'], updp_p['a'], updp_p['p'])
    loc_param0, sca_param0, updp_m['b'], updp_m['a'], updp_m['p'] = capfloor_params(lim_params, loc_param, sca_param,updp_m['b'], updp_m['a'], updp_m['p'])

    dataset_in_pdf = genhyperbolic.pdf( dataset_in, loc=loc_param, scale=sca_param, b=b_param, a=a_param , p=p_param )
    grad_single =   genhyperbolic.pdf( dataset_in, loc=loc_param, scale=sca_param, b=updp_p['b'], a=updp_p['a'], p=updp_p['p'] ) - \
                    genhyperbolic.pdf( dataset_in, loc=loc_param, scale=sca_param, b=updp_m['b'], a=updp_m['a'], p=updp_m['p']  )
    grad_single /= (2*h)
    grad_single = -np.sum( grad_single /dataset_in_pdf  )  / len(dataset_in)

    del dataset_in_pdf; del loc_param; del sca_param; del b_param; del a_param; del p_param; del sp; del dataset_in; del consider_skewness; del consider_nonzero_p; del lim_params

    return grad_single

#------------------------------------------------------------------------------------------------
'''
    print("CODA:")
    loc_param1=0.0015705261418902002; sca_param1=0.011285118279652733;
    b_param1= -0.01 #-0.03905225263562642
    a_param1=0.4693707945824144; p_param1= - 0.08358465526128243
    for i in range(10):
        print(i, ") ", - (np.sum( np.log( genhyperbolic.pdf(dataset_in, loc=loc_param1, scale=sca_param1, b=b_param1*i, a=a_param1, p=p_param1 ) ) ))/len(dataset_in) )
    exit(0)
'''