'''
The functions of this module fit a given dataset ( << dataset_in >> ) to a t-student function.

Suggestion (2021-10-12): set: tolerance_fitting = 0.000005; max_n_iter = 150; n_random_tries = 150; step = uniform(0, 20).
                              starting guess: [ np.median(dataset_in) ] ; [ sca_guess, sca_guess/8 ] ; [0] ... ; [ 2.5 ]
'''


from scipy.stats import levy_stable, cauchy
import numpy as np
from math import isnan
np.seterr(invalid='ignore') # This tells NumPy to hide any warning with some “invalid” message in it.
np.seterr(divide='ignore')
from random import uniform
from module_fitting import first_iteration, first_iteration_single_param
from module_parameters import tolerance_fitting
#np.random.seed(seed=1234366); random.seed(1033)

#------------------------------------------------------------------------------------------------


''' 
np.random.seed(seed=1234366)
myalpha = 0.1; mybeta=0.2
print( levy_stable.rvs( alpha=myalpha, beta=mybeta, size=10 ))
#print( levy_stable.rvs( myalpha, mybeta, size=10))
exit(0)
'''

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
            self.skew_dataset_in = skew(dataset_in)
            self.max_allowed_abs_skewparam = min( 2*abs(self.skew_dataset_in), 0.75 ) # In stable distributions the skewness parameter (beta) must be in [-1,1].
        else:
            self.max_allowed_abs_skewparam = 0;
        self.min_allowed_abs_a = 1.01;    # In stable distributions the stability parameter (alpha) must be in (0, 2].  2 corresponds to normal distribution, and 1 corresponds to Cauchy. We cap the possible values to be less fat-tailed than Cauchy.
        self.max_allowed_abs_a = 2;

#------------------------------------------------------------------------------------------------

def find_starting_point( dataset_in, consider_skewness = True, res_nct={'nct_loc':float('NaN')}):
    ''' This function determines the values of the starting point for the optimization algorithm.

    :param dataset_in: (numpy array of floats): The dataset whose fitting to a t-student function is sought.
           IMPORTANT: It must be sorted !!!
    :param consider_skewness: (Boolean) False if the b parameter to consider is zero (symmetric distribution); True otherwise.
    :return: (lists of numbers) li_locations_sg, li_scalings_sg, li_beta_params_sg, li_alpha_params_sg, li_s_sg
    '''

    from scipy.stats import skew

    myvar = dataset_in[int(len(dataset_in) / 4): int(3 *len(dataset_in) / 4)]
    sca_guess = 2*np.std(myvar)
    skewness_dataset = skew( dataset_in )
    print("The skewness of the time series is:",skewness_dataset)

    if not ( isnan(res_nct['nct_loc']) ): # We read the starting point from the results of nct (if they exist and it is required)
        li_locations_sg = [ np.median(dataset_in) ]
        li_scalings_sg  = [ 0.757*res_nct['nct_scale']+0.00031 ]
        if ( consider_skewness == False ):
           li_beta_params_sg = [0]
        else:
           li_beta_params_sg = [  np.sign(res_nct['nct_skparam'])*min( 0.9*abs(res_nct['nct_skparam']), 0.35)  ]
           #if (np.sign(skewness_dataset)!=np.sign(res_nct['nct_skparam'])):  li_beta_params_sg.append( 0.6 / (1 + np.exp(-4*skewness_dataset)) - 0.3 ) # This line does not seem to help. The sign of alpha_param is usually the same as the sign of the nct-skewness-param, even if it is opposite to the sign of the skewness of the data.
        li_alpha_params_sg = [ min(0.11*res_nct['nct_dfparam']+1.26,1.8) ]
    else:
        li_locations_sg =  [ np.median(dataset_in) ]
        li_scalings_sg  =  [ sca_guess ]
        if ( consider_skewness == False ):
           li_beta_params_sg = [0]
        else:
            li_beta_params_sg = [ 0.6 / (1 + np.exp(-4*skewness_dataset)) - 0.3 ]
        li_alpha_params_sg = [ 1.68 ] # 1 corresponds to Cauchy; 2 corresponds to Gaussian. We choose these values because we check that our functions (single sweep of 'a' param) can easily change it from 1.31 to 1.51 in one single iteration.

    del dataset_in; del consider_skewness; del myvar; del skewness_dataset;

    #li_locations_sg = [0]; li_scalings_sg = [0.012];  li_beta_params_sg=[0]; li_alpha_params_sg = [1.63]

    return li_locations_sg, li_scalings_sg, li_beta_params_sg, li_alpha_params_sg

#------------------------------------------------------------------------------------------------

def update_optimal_parameters( loss_fnd, loc_param_fnd, sca_param_fnd, beta_param, alpha_param_fnd, loss_opt, loc_param_opt, sca_param_opt, beta_param_opt, alpha_param_opt):
    '''This function updates the values of the parameters of the fitting'''
    if ( (loss_fnd < loss_opt) or (loc_param_opt == None) ):
        del loss_opt; del loc_param_opt; del sca_param_opt; del beta_param_opt; del alpha_param_opt;
        return loss_fnd, loc_param_fnd, sca_param_fnd, beta_param, alpha_param_fnd
    else:
        del loss_fnd; del loc_param_fnd; del sca_param_fnd; del beta_param; del alpha_param_fnd;
        return loss_opt, loc_param_opt, sca_param_opt, beta_param_opt, alpha_param_opt

#------------------------------------------------------------------------------------------------

def fit_to_stable_global_minimum( dataset_in,  n_random_tries,  max_n_iter, consider_skewness=True, verbose=0, res_nct=None ):
    ''' This function finds the generalized hyperbolic distribution which best fits to the input dataset.
    The suffixes of variables below mean: # "_tt" means "to be tried"; "_opt" means "optimal"; "_fnd" means "found"

    :param dataset_in: (numpy array of floats): The dataset whose fitting to a t-student function is sought.
                       IMPORTANT: It must be sorted !!!
    :param consider_skewness: (Boolean) False if the skewness parameter to consider is zero (symmetric distribution); True otherwise.
    :param : (Boolean) False if the p parameter to consider is zero; True otherwise.
    :return: (4 float numbers) nct_location, nct_scaling, nct_skewparam, nct_df
    '''

    li_locations_sg, li_scalings_sg, li_b_sg, li_a_sg = find_starting_point( dataset_in, consider_skewness, res_nct )
    lim_params = LimitsForParams(dataset_in, consider_skewness)

    loss_opt = 99999999; loc_param_opt = None; sca_param_opt = None; beta_param_opt = None; alpha_param_opt = None;

    for loc_param_tt in li_locations_sg:
        for sca_param_tt in li_scalings_sg:
            for beta_param_tt in li_b_sg:
                for alpha_param_tt in li_a_sg:
                      print("Params IN", loc_param_tt, sca_param_tt, beta_param_tt, alpha_param_tt ) #if (verbose>0):
                      for trial_counter in range(n_random_tries):

                          loss_fnd, loc_param_fnd, sca_param_fnd, beta_param, alpha_param_fnd  = fit_to_levy_stable_local_minimum( dataset_in, max_n_iter, consider_skewness, lim_params, loc_param_tt, sca_param_tt, beta_param_tt, alpha_param_tt, verbose )
                          loss_opt, loc_param_opt, sca_param_opt, beta_param_opt, alpha_param_opt = update_optimal_parameters( loss_fnd, loc_param_fnd, sca_param_fnd, beta_param, alpha_param_fnd,
                                                                                                                                      loss_opt, loc_param_opt, sca_param_opt, beta_param_opt, alpha_param_opt)
                          print("Params IN",  loc_param_tt, sca_param_tt, beta_param_tt, alpha_param_tt , "; Params OUT", loc_param_fnd, sca_param_fnd, beta_param, alpha_param_fnd, "; LOSS:", loss_fnd )

    print("\n The GLOBAL minimum (stable) is:", loc_param_opt,  sca_param_opt, beta_param_opt,  alpha_param_opt,"; Loss:", loss_opt,"\n" )

    del dataset_in; del n_random_tries; del  max_n_iter; del consider_skewness

    return { 'distribution_type':'levy_stable', 'loc_param': loc_param_opt, 'scale_param':sca_param_opt, 'beta_param':beta_param_opt, 'alpha_param':alpha_param_opt, 'loss':loss_opt }

#------------------------------------------------------------------------------------------------

def capfloor_params(lim_params,  loc_param1, sca_param1, beta_param1, alpha_param1 ):

    loc_param1 = np.sign(loc_param1) * min(abs(loc_param1), lim_params.max_allowed_abs_loc  )
    sca_param1 = max(sca_param1, lim_params.min_allowed_scale); sca_param1 = min(sca_param1, lim_params.max_allowed_scale);
    beta_param1 = np.sign(beta_param1) * min( abs( beta_param1 ), lim_params.max_allowed_abs_skewparam )
    alpha_param1 = np.sign(alpha_param1) * min( abs(alpha_param1), lim_params.max_allowed_abs_a)
    alpha_param1 = np.sign(alpha_param1) * max( abs(alpha_param1), lim_params.min_allowed_abs_a)

    # The generalized hyperbolic distribution demands that |a|>=|b|, otherwise complex numbers appear. We modify a because b varies slowly.
    #if ( abs(beta_param1) >= abs(alpha_param1) ):
    #    alpha_param1 *= uniform(1.1,4) * abs(beta_param1)/ ( abs(alpha_param1) )

    del lim_params

    return loc_param1, sca_param1, beta_param1, alpha_param1

#------------------------------------------------------------------------------------------------

def fit_to_levy_stable_local_minimum( dataset_in, max_n_iter, consider_skewness, lim_params, loc_param0, sca_param0, beta_param0, alpha_param0, verbose=0  ):
    ''' This function finds the generalized hyperbolic distribution which best{ fits to the input dataset.

    The suffixes of variables below mean: # "_tt" means "to be tried"; "_opt" means "optimal"; "_fnd" means "found"

    :param dataset_in: (numpy array of floats) set of values whose pdf is to be fit.
    :param consider_skewness: (Boolean) False if the skewness parameter to consider is zero (symmetric distribution); True otherwise.
    :param : (Boolean) False if the p parameter to consider is zero; True otherwise.
    :return: loss_opt, loc_param_opt, sca_param_opt, beta_param_opt, alpha_param_opt,
    '''

    #np.random.seed(seed=1234366); random.seed(1033)

    # Initialization
    loss_opt = 99999999; loc_param_opt = None; sca_param_opt = None; beta_param_opt = None; alpha_param_opt = None;
    metropolis = 0;
    list_single_sweep = [];
    if (consider_skewness): list_single_sweep.append('b')
    list_single_sweep.append('a')

    # First iteration (which involves a random choice of the other point to start the Barzilai-Borwein process)
    loss0, loc_param0, sca_param0, beta_param0, alpha_param0, trash1, grad_loc0, grad_sca0, grad_beta0, grad_alpha0, trash2, loss1, loc_param1, sca_param1, beta_param1, alpha_param1, trash3 = first_iteration(dataset_in, 'levy_stable', consider_skewness, loc_param0, sca_param0, beta_param0, alpha_param0, None, False, verbose )
    loss_opt, loc_param_opt, sca_param_opt, beta_param_opt, alpha_param_opt = update_optimal_parameters(loss1,loc_param1,sca_param1,beta_param1,alpha_param1,loss0,loc_param0,sca_param0,beta_param0,alpha_param0)

    if (None in [loc_param0, sca_param0, beta_param0, alpha_param0]):
        del dataset_in; del max_n_iter; del consider_skewness; del lim_params;  del loc_param0; del sca_param0; del alpha_param0; del beta_param0; del verbose; del trash1; del trash2; del trash3
        return 9999, None, None, None, None
    loc_param1, sca_param1, beta_param1, alpha_param1 = capfloor_params(lim_params,  loc_param1, sca_param1, beta_param1, alpha_param1 )
    #loc_param1 = np.sign(loc_param1) * min(abs(loc_param1), lim_params.max_allowed_abs_loc)

    # Sweeping to find the local minimum
    n_iter=0
    while ( (n_iter < max_n_iter/2) or ( ( (( abs(loss1 - loss0) > tolerance_fitting) or (metropolis > 0.9) ) or (loss1 == 9999) ) and (n_iter < max_n_iter) )):

        # Update the loss to check convergence
        n_iter += 1
        loss0 = loss1; metropolis = uniform(0, 1)

        # Find the gradient
        grad_loc1, grad_sca1, grad_beta1, grad_alpha1 = calculate_gradient_params( dataset_in, consider_skewness, loc_param1, sca_param1, beta_param1, alpha_param1 )

        # Find the step size
        step = ( (loc_param1 - loc_param0) * (grad_loc1 - grad_loc0) + (sca_param1 - sca_param0) * (grad_sca1 - grad_sca0)  + \
               (beta_param1 - beta_param0) * (grad_beta1 - grad_beta0) + (alpha_param1 - alpha_param0) * (grad_alpha1 - grad_alpha0) )
        step /= -( (grad_loc1 - grad_loc0)**2 + (grad_sca1 - grad_sca0)**2 + (grad_beta1 - grad_beta0)**2 + (grad_alpha1 - grad_alpha0)**2   )

        # Update quantities
        loc_param0 = loc_param1;   sca_param0 = sca_param1;  beta_param0 = beta_param1;  alpha_param0 = alpha_param1;
        grad_loc0  = grad_loc1   ; grad_sca0  = grad_sca1   ; grad_beta0 = grad_beta1;    grad_alpha0 = grad_alpha1;
        loc_param1  += step * grad_loc1;  sca_param1  += step * grad_sca1;  beta_param1 += step * grad_beta1;  alpha_param1 += step * grad_alpha1;
        loc_param1, sca_param1, beta_param1, alpha_param1 = capfloor_params(lim_params, loc_param1, sca_param1, beta_param1, alpha_param1)

        # Find the loss
        try:
            loss1 = - (np.sum( np.log( levy_stable.pdf(dataset_in, loc=loc_param1, scale=sca_param1, beta=beta_param1, alpha=alpha_param1) ) ))/len(dataset_in)
            if ((loss1 > -2) or isnan(loss1)):
                loc_param1 = uniform(-0.0001,0.0001); sca_param1 = uniform(0.001,0.01); beta_param1 = 0 ; alpha_param1 = uniform(1.5,1.8)
        except RuntimeWarning:
            loc_param1 = uniform(-0.0001,0.0001); sca_param1 = uniform(0.001,0.01); beta_param1 = 0 ; alpha_param1 = uniform(1.5,1.8)
        loss_opt, loc_param_opt, sca_param_opt, beta_param_opt, alpha_param_opt = update_optimal_parameters(loss1,loc_param1,sca_param1,beta_param1,alpha_param1,loss_opt,loc_param_opt,sca_param_opt,beta_param_opt,alpha_param_opt)
        if ((abs(loss1 - loss0) < tolerance_fitting) and (n_iter < max_n_iter / 2)):
            loc_param1 *= 0.9; sca_param1*=uniform(0.75,1.5); beta_param1*=0.9; alpha_param1 += uniform(-0.15,0.15)

        if (verbose > 1): print("iter ", n_iter, ") Params", loc_param1, sca_param1, beta_param1, alpha_param1,"; loss=", loss1)

        if  ((n_iter%3)==0):
            for sp in list_single_sweep:
                loss1,loc_param1,sca_paramx,beta_param1,alpha_param1 = fit_to_levy_stable_local_minimum_sweep_single_param(sp, dataset_in, consider_skewness, lim_params, loc_param1,sca_param1,beta_param1,alpha_param1, verbose )
                loss_opt, loc_param_opt, sca_param_opt, beta_param_opt, alpha_param_opt = update_optimal_parameters(loss1,loc_param1,sca_param1,beta_param1,alpha_param1,loss_opt,loc_param_opt,sca_param_opt, beta_param_opt,alpha_param_opt)


    del dataset_in; del max_n_iter; del consider_skewness; del lim_params; del loc_param0; del sca_param0; del beta_param0; del alpha_param0

    return loss_opt, loc_param_opt, sca_param_opt, beta_param_opt, alpha_param_opt

#----------------------------------------------------------------------------------------------------------------------

def fit_to_levy_stable_local_minimum_sweep_single_param( sp, dataset_in, consider_skewness, lim_params, loc_param0, sca_param0, beta_param0, alpha_param0, verbose ):
    '''
    DEV: On 2022-12-04 I check that this function can strongly change the sweeped parameter (either the skewness parameter 'b'
    or the tail parameter 'a'. For example, 'a' changes from 1.31 to 1.51 (or from 1.51 to 1.64) in one single step, and
    'b' change from -0.2578 to -0.1406 (or from -0.13 to -0.87) in one single step.
    '''

    from random import uniform
    from math import isnan

    # Initialization
    loss0 = 99; loss1 = 999;  loc_param_opt = loc_param0; sca_param_opt = sca_param0; beta_param_opt = beta_param0; alpha_param_opt = alpha_param0;
    loss_opt = - (np.sum(np.log(levy_stable.pdf(dataset_in, loc=loc_param0, scale=sca_param0, beta=beta_param0, alpha=alpha_param0)))) / len(dataset_in)
    metropolis = 0;

    # First iteration
    updp0, grad_sp0, updp1 = first_iteration_single_param( sp, dataset_in, 'levy_stable', consider_skewness,  lim_params, loc_param0, sca_param0, beta_param0, alpha_param0  )
    if (None in [updp0, grad_sp0, updp1]):
        del dataset_in; del lim_params;  del loc_param0; del sca_param0; del beta_param0; del alpha_param0
        return loss_opt, loc_param_opt, sca_param_opt, beta_param_opt, alpha_param_opt

    loc_param0, sca_param0, updp1['b'], updp1['a']  = capfloor_params(lim_params, loc_param0, sca_param0, updp1['b'],updp1['a'] )

    # Sweeping to find the local minimum
    max_n_iter = 8
    n_iter=0
    while ( ( (( abs(loss1 - loss0) > tolerance_fitting) or (metropolis > 0.9) ) or (loss1 == 9999) ) and (n_iter < max_n_iter) ):

        n_iter += 1
        loss0 = loss1; metropolis = uniform(0, 1)

        grad_sp1 = calculate_gradient_single_param(sp, dataset_in, consider_skewness, lim_params, loc_param0, sca_param0, updp1['b'] , updp1['a']  )
        step = - (updp1['b']-updp0['b']+updp1['a']-updp0['a'] ) * (grad_sp1 - grad_sp0) / ( (grad_sp1 - grad_sp0)**2 )
        updp0    = updp1.copy()
        grad_sp0 = grad_sp1
        updp1[sp] += step * grad_sp1
        #print("    ---- step=", step,"grad=",grad_sp1, "updated=", updp1[sp])

        loc_param0, sca_param0, updp1['b'], updp1['a']  = capfloor_params(lim_params, loc_param0, sca_param0, updp1['b'],updp1['a'])

        # Find the loss
        try:
            loss1 = - (np.sum( np.log( levy_stable.pdf(dataset_in, loc=loc_param0, scale=sca_param0, beta=updp1['b'], alpha=updp1['a']  ) ) ))/len(dataset_in)
            if (verbose > 1): print("    --- mini-iter: ",n_iter,"Params: loc=",loc_param0, "scale",sca_param0,"b",updp1['b'], "a",updp1['a'],"; loss=",loss1 )
            if ((loss1 > -2) or isnan(loss1)):
                break
        except RuntimeWarning:
            break #loc_param1 = uniform(-1,1); sca_param1 = uniform(0.2,2); beta_param1 = 0 ; alpha_param1 = uniform(2,4); 1 = 0

        loss_opt, loc_param_opt, sca_param_opt, beta_param_opt, alpha_param_opt = update_optimal_parameters(loss1,loc_param0,sca_param0,updp1['b'],updp1['a'], loss_opt,loc_param_opt,sca_param_opt,beta_param_opt,alpha_param_opt)
        # if (n_iter == max_n_iter-1): print("  WARNING: Maximum number of iterations reached.")

    del dataset_in; del max_n_iter; del lim_params;  del loc_param0; del sca_param0; del beta_param0; del alpha_param0

    return  loss_opt, loc_param_opt, sca_param_opt, beta_param_opt, alpha_param_opt

#------------------------------------------------------------------------------------------------

def fit_to_levy_stable_local_minimum_sweep_beta( dataset_in, lim_params, loc_param0, sca_param0, beta_param0, alpha_param0, loss0  ):
    ''' This function sweeps the values of the skewness parameter using a equipaced grid. This is found to be more effective
    (for the b parameter) than either doing gradient descent in all 5 parameters or in the b parameter alone (in both latter
    cases the b parameter hardly changes).
    Example:
        iter  1 ) Params 0.0013263767625094906 0.011543424896024449 0.1593836498447446 0.6822689618860058 -0.008124297673654872 ; loss= -2.4817292105199233
        Optimal prev = 0.1593836498447446 Loss -2.4817292105199233
        Optimal found= -0.04722265141522439 Loss -2.53304951629618
    '''

    loss_opt = loss0; loc_param_opt=loc_param0; sca_param_opt=sca_param0;  beta_param_opt=beta_param0;alpha_param_opt=alpha_param0;
    if (abs(lim_params.skew_dataset_in))>0.1:   n_essais = 50
    elif (abs(lim_params.skew_dataset_in))>0.08: n_essais = 40
    elif (abs(lim_params.skew_dataset_in)) > 0.05: n_essais = 30
    else: n_essais = 20

    for i in range(n_essais):
        my_beta = lim_params.skew_dataset_in * (i- int(n_essais/4)) / int(n_essais/2)
        loss1 = - (np.sum(np.log(levy_stable.pdf(dataset_in, loc=loc_param0, scale=sca_param0, beta=my_beta, alpha=alpha_param0)))) / len(dataset_in)
        loss_opt, loc_param_opt, sca_param_opt, beta_param_opt, alpha_param_opt = update_optimal_parameters(loss1,loc_param0,sca_param0,my_beta, alpha_param0,loss_opt, loc_param_opt,sca_param_opt,beta_param_opt,alpha_param_opt)

    #print("Optimal found=",beta_param_opt,"Loss",loss_opt)

    del dataset_in; del lim_params;  del loc_param0; del sca_param0;  del alpha_param0; del my_beta; del n_essais

    return  loss_opt, beta_param_opt

#------------------------------------------------------------------------------------------------

def calculate_gradient_params( dataset_in, consider_skewness, loc_param, sca_param, beta_param, alpha_param):
    ''' This function retunrs the gradient of the parameters (4-component array). Note that the formula is based
    on the maximum likelihood method. The loss function is defined as minus the log of the products of all the
    values (pdf's of all points x of datasetin), this is the sum of the logs of each pdf (-sum_i(log(pdf(x_i)))
    and hence the derivative of it wrt the 4 parameters is f'(y)/f(y), where y:=pdf(x_i).

    :param dataset_in: (numpy array of floats) set of values whose pdf is to be fit.
    :param consider_skewness: (Boolean) False if the skewness parameter to consider is zero (symmetric distribution); True otherwise.
    :param : (Boolean) False if the p parameter to consider is zero; True otherwise.
    :param loc_param: (float) location parameter
    :param sca_param: (float) scaling parameter
    :param beta_param: (float) skewness parameter
    :param alpha_param: (float) a parameter
    :param : (float) p parameter
    :return: (5 float numbers) grad_loc, grad_sca, grad_beta, grad_alpha, grad_p
    '''

    #h0 = sca_param  / 100000
    #h1 = beta_param / 100 + 1/1000
    #h2 = alpha_param   / 100000

    h0 = sca_param / 10000
    h1 = beta_param / 100 + 1 / 1000
    h2 = alpha_param / 100

    dataset_in_pdf = levy_stable.pdf( dataset_in, loc=loc_param, scale=sca_param, beta=beta_param, alpha=alpha_param )

    grad_loc = levy_stable.pdf( dataset_in, loc=loc_param+h0, scale=sca_param, beta=beta_param, alpha=alpha_param) - \
               levy_stable.pdf( dataset_in, loc=loc_param-h0, scale=sca_param, beta=beta_param, alpha=alpha_param)
    grad_loc /= (2*h0)
    grad_loc = -np.sum( grad_loc / dataset_in_pdf  )  / len(dataset_in)

    grad_sca = levy_stable.pdf( dataset_in, loc=loc_param, scale=sca_param+h0, beta=beta_param, alpha=alpha_param) - \
               levy_stable.pdf( dataset_in, loc=loc_param, scale=sca_param-h0, beta=beta_param, alpha=alpha_param)
    grad_sca /= (2*h0)
    grad_sca = -np.sum( grad_sca / dataset_in_pdf  )  / len(dataset_in)

    if consider_skewness:
       grad_beta = levy_stable.pdf( dataset_in, loc=loc_param, scale=sca_param, beta=beta_param+h1, alpha=alpha_param) - \
                levy_stable.pdf( dataset_in, loc=loc_param, scale=sca_param, beta=beta_param-h1, alpha=alpha_param)
       grad_beta /= (2*h1)
       grad_beta = -np.sum( grad_beta /dataset_in_pdf  )  / len(dataset_in)
    else:
       grad_beta = 0

    grad_alpha =   levy_stable.pdf( dataset_in, loc=loc_param, scale=sca_param, beta=beta_param, alpha=alpha_param+h2) - \
               levy_stable.pdf( dataset_in, loc=loc_param, scale=sca_param, beta=beta_param, alpha=alpha_param-h2  )
    grad_alpha /= (2*h2)
    grad_alpha = -np.sum( grad_alpha /dataset_in_pdf  )  / len(dataset_in)

    return grad_loc, grad_sca, grad_beta, grad_alpha

#------------------------------------------------------------------------------------------------

def calculate_gradient_single_param( sp, dataset_in, consider_skewness, lim_params, loc_param, sca_param, beta_param, alpha_param):
    '''   sp (single param, whose gradient is calculated) must be either 'a', 'b'.
    '''

    if ( (not consider_skewness) and (sp=='b') ): return 0

    if   (sp== 'a'):
        h = alpha_param   / 200 + 1/1000
        updp_p = { 'a': alpha_param+h, 'b':beta_param } # Updated parameters (plus and minus)
        updp_m = { 'a': alpha_param-h, 'b':beta_param }
    elif (sp== 'b'):
        h = beta_param / 200 + 1/1000
        updp_p = {'a': alpha_param, 'b': beta_param + h }
        updp_m = {'a': alpha_param, 'b': beta_param - h }

    loc_param0, sca_param0, updp_p['b'], updp_p['a']  = capfloor_params(lim_params, loc_param, sca_param, updp_p['b'], updp_p['a']  )
    loc_param0, sca_param0, updp_m['b'], updp_m['a']  = capfloor_params(lim_params, loc_param, sca_param,updp_m['b'], updp_m['a']  )

    dataset_in_pdf = levy_stable.pdf( dataset_in, loc=loc_param, scale=sca_param, beta=beta_param, alpha=alpha_param )
    grad_single =   levy_stable.pdf( dataset_in, loc=loc_param, scale=sca_param, beta=updp_p['b'], alpha=updp_p['a'] ) - \
                    levy_stable.pdf( dataset_in, loc=loc_param, scale=sca_param, beta=updp_m['b'], alpha=updp_m['a'] )
    grad_single /= (2*h)
    grad_single = -np.sum( grad_single /dataset_in_pdf  )  / len(dataset_in)

    del dataset_in_pdf; del sp; del dataset_in; del loc_param; del sca_param; del beta_param; del alpha_param

    return grad_single

#------------------------------------------------------------------------------------------------

