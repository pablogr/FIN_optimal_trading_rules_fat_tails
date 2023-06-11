''' This module generates random walks using independent (binomial) and non-independent variations (abs returns which depend on the price itself). The aim of this is to check that non-independent variations generate fat-tailed distributions.
The initial price is 1.
'''

import numpy as np
import pandas as pd
from numpy import exp
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, norm

# -----------------------------------------------------------------------------------------------

def analysis_random_walks(probab=0.5, Npaths=100000, Njumpsperpath=2000):
    '''This function calculates the distribution of prices after Njumpsperpath binomial random variations.
    This is the function used to perform the calculations of the first section of the SI of the paper.
    Use the function uniform for probabilities different from 0.5.
    '''

    from numpy.random import uniform, randint
    from module_random_walks import generate_binomial_random_walk, generate_conditional_random_walk

    if ( abs(probab-0.5)<0.00000000001):
        array_uniform = randint(2, size=Njumpsperpath * Npaths)
    else:
        array_uniform = uniform(0, 1, Njumpsperpath*Npaths )

    generate_binomial_random_walk(array_uniform, probab, Npaths, Njumpsperpath)
    generate_conditional_random_walk(array_uniform, probab, Npaths, Njumpsperpath, False, 6)
    generate_conditional_random_walk(array_uniform, probab, Npaths, Njumpsperpath, True, 12)

    '''EXAMPLE OUTPUTS:

    Npaths = 1M,  Njumpsperpath =2k
    The Kurtosis of the binomial with constant variation is -0.002166211861612588
    Loc= 1.0000642120000005 ; sigma= 0.022388864929221975
    The Kurtosis of the conditional is 0.6392684519129217
    Loc= 1.006274087426469 ; sigma= 2.7259602648987697
    The Kurtosis of the conditional is 1.5233488638948973
    Loc= 1.0000743472627558 ; sigma= 0.026018258566829818

    Npaths = 3M,  Njumpsperpath = 2k:
    The Kurtosis of the binomial with constant variation is 0.0010580986500214884
    Loc= 0.9999690669999993 ; sigma= 0.02235237097377818
    The Kurtosis of the conditionalasymmetrical is 0.6512030002946645
    Loc= 0.9967545007689379 ; sigma= 2.7225791437857914
    The Kurtosis of the conditionalsymmetric is 1.5433987989808982
    Loc= 0.9999657301211144 ; sigma= 0.025966965315349606
    '''

    return

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def generate_binomial_random_walk( array_uniform, probab=0.5, Npaths=5000, Njumpsperpath=500 ):

    variation = 1/Njumpsperpath
    results = np.empty([Npaths])
    if (abs(probab - 0.5) < 0.00000000001):

        for pathcount in range(Npaths):
            p = 1 # p stands for "price"
            for i in range(Njumpsperpath):
                if (array_uniform[pathcount * Njumpsperpath + i] == 0):
                    p -= variation # lognormal: p *= (1 - variation)
                else:
                    p += variation # lognormal: p *= (1 +  variation)
                #print(i,"(",array_uniform[ pathcount*Njumpsperpath +i],probab,")",p)
            results[pathcount] = p

    else: #if ( abs(probab-0.5)<0.00000000001):

        for pathcount in range(Npaths):
            p = 1 # p stands for "price"
            for i in range(Njumpsperpath):
                if (  array_uniform[ pathcount*Njumpsperpath +i]   <= probab ):
                    p -= variation
                else:
                    p += variation
            results[pathcount] = p

    print("The Kurtosis of the binomial with constant variation is",kurtosis(results))
    #print(results);
    plot_histogram( results, "Constant variation" )

    return

# -----------------------------------------------------------------------------------------------

def generate_conditional_random_walk( array_uniform, probab0=0.5, Npaths=5000, Njumpsperpath=500, symmetric_variations=True, a=12 ):

    variation0 = 1/Njumpsperpath
    results = np.zeros([Npaths])

    if (abs(probab0 - 0.5) < 0.00000000001):

        for pathcount in range(Npaths):
            p = 1 # p stands for "price"
            for i in range(Njumpsperpath):

                variation = variation0 * (1 + a * abs(p - 1))

                if (symmetric_variations):
                    #probab = probab0 - ((1-probab0)/a)*((exp(a*(p-1))-exp(-a*(p-1)))/(exp(a*(p-1))+exp(-a*(p-1))))

                    if (  array_uniform[ pathcount*Njumpsperpath +i]   == 0 ):
                        p -= variation  # lognormal: p *= (1 - variation)
                    else:
                        p += variation  # lognormal: p *= (1 +  variation)
                else: # The price tends to increase more than to decrease if it is positive, and the converse if it is negative.

                    if (p>1):
                        if (array_uniform[pathcount * Njumpsperpath + i] == 0):
                            p -= variation0
                        else:
                            p += variation
                    else:
                        if (array_uniform[pathcount * Njumpsperpath + i] == 0):
                            p -= variation
                        else:
                            p += variation0
                if (p<=0):
                    p=0
                    break
            results[pathcount] = p

    else: #if ( abs(probab-0.5)<0.00000000001):

        for pathcount in range(Npaths):
            p = 1

            for i in range(Njumpsperpath):

                variation = variation0 * (1 + a * abs(p - 1))

                if (symmetric_variations):

                    if (  array_uniform[ pathcount*Njumpsperpath +i]   <= probab0 ):
                        p -= variation
                    else:
                        p += variation

                else:  # Asymmetrical variations: The price tends to increase more than to decrease if it is positive, and the converse if it is negative.

                    if (p > 1):
                        if (array_uniform[pathcount * Njumpsperpath + i] <= probab0 ):
                            p -= variation0
                        else:
                            p += variation
                    else:
                        if (array_uniform[pathcount * Njumpsperpath + i] <= probab0 ):
                            p -= variation
                        else:
                            p += variation0
                if (p <= 0):
                    p = 0
                    break
            results[pathcount] = p


    #print(results)
    if (symmetric_variations): text="symmetric"
    else: text = "asymmetrical"
    print("The Kurtosis of the conditional" + text + " is", kurtosis(results))
    plot_histogram( results, "Variable "+text+" variation" )

    return


# -----------------------------------------------------------------------------------------------

def plot_histogram(  array_in, name ):
    ''' This function plots a histogram.
    '''

    sigma = np.std(array_in)
    loc   = np.mean(array_in)
    array_in = (array_in - loc) / sigma
    print("Loc=",loc,"; sigma=",sigma)

    # Define the bins
    '''
    len_to_remove = int(0.01 * len(array_in))
    lim_bins = 0.4 #4*np.std( array_in[len_to_remove:-len_to_remove] )
    num_bins = 31
    bins = [1-lim_bins  ] # OLD: [-lim_bins + loc_in ]; Si el loc se calculó mal, esto hace que el histograma tb se pinte mal.
    labels = []
    for i in range(1,num_bins+1):
        bins   += [ 1-lim_bins + i*(2*lim_bins)/num_bins ] #old:  [ -lim_bins + loc_in  + i*(2*lim_bins)/num_bins ]
        if (i%8 == 0):
          if (lim_bins >= 1):
            labels += [ "{:.4f}".format( -lim_bins  + (i-0.5)*(2*lim_bins)/num_bins  ) ] # old: labels += [ "{:.4f}".format( -lim_bins + loc_in  + (i-0.5)*(2*lim_bins)/num_bins  ) ]
          else:
            labels += ["{:.5f}".format(-lim_bins  + (i - 0.5) * (2 * lim_bins) / num_bins)] # old:  ["{:.5f}".format(-lim_bins + loc_in  + (i - 0.5) * (2 * lim_bins) / num_bins)]
        else:
            labels += [None]
    '''
    delt = 0.5
    bins = np.arange(-4-delt/2,4+delt/2+0.000001,delt)

    # Calculate values of the "float" histogram
    counts0, bins = np.histogram( array_in, bins )
    counts = np.zeros(len(counts0))
    for i in range(len(counts0)):
        #print(bins[i],bins[i+1]); print(i, counts0[i])
        counts[i] = counts0[i]/len(array_in)

    #for i in range(len(counts0)-1):print(bins[i],bins[i+1],":",counts[i])

    # Plot the histogram
    x = np.array( [ (bins[i]+bins[i+1])/2 for i in range(len(bins)-1) ] ) # old: np.arange(len(labels))

    width = 0.91*(bins[i+1]-bins[i])
    fig, ax = plt.subplots()
    ax.bar(x , counts, width, label='Histogram (normalized)', color="darkblue") #'mediumblue'

    # Plot fitting function:
    x_cont = array_in
    num_bins = len(bins)-1
    x_cont0 = np.array([x[0] + i * (x[-1] - x[0]) / (12 * num_bins - 1) for i in range(12 * num_bins)])  # np.linspace(0, len(labels) - 1, num=num_bins*10  )
    loc_in = np.mean(array_in)
    scale_in = np.std(array_in)
    y_cont0 = norm.pdf(x_cont0, loc=loc_in, scale=scale_in) * (bins[i + 1] - bins[i])
    #y_cont = norm.pdf(x_cont, loc=loc_in, scale=scale_in) * (bins[i + 1] - bins[i])
    mycolor = "red"
    #ax.plot(x_cont0, y_cont0, color='black', lw=5.2)
    #ax.plot(x_cont, y_cont, '.', color='black', markersize=17)
    ax.plot(x_cont0, y_cont0, label='Fit to normal', color=mycolor, lw=2)

    ''' 
    for fitting_function in list_sweep:
        #loc_plot = (len(labels)-1)/2
        #scale_plot = scale_in * len(labels) / (2*lim_bins)
        x_cont = x
        x_cont0 = np.array( [ x[0] + i*(x[-1]-x[0])/(12*num_bins-1) for i in range(12*num_bins) ]  ) #np.linspace(0, len(labels) - 1, num=num_bins*10  )
        name_plot = fitted_time_series.filename; name_plot = name_plot.replace(".csv","")
        
        
        loc_in, scale_in, skew_in, tail1_in, tail2_in = provide_curve_parameters(fitting_function, fitted_time_series, plot_several_curves, params_several_curves)
        if (None in [loc_in, scale_in]): continue
        if ( fitting_function =="norm"):
           y_cont0 = norm.pdf(x_cont0, loc=loc_in, scale=scale_in ) * (bins[i + 1] - bins[i] )
           y_cont  = norm.pdf(x_cont,  loc=loc_in, scale=scale_in ) * (bins[i + 1] - bins[i] )
           mycolor  = "red"
           ax.plot(x_cont0, y_cont0, color='black', lw=5.2)
           if not (plot_several_curves): ax.plot(x_cont, y_cont, '.', color='black', markersize=17)
           ax.plot(x_cont0, y_cont0, label='Fit to normal', color=mycolor, lw=3)
           pathout = f"{fitted_time_series.directory_output_plots}/histofit_{name_plot}_normal.pdf"
        elif (fitting_function == "nct"):
           y_cont0 = nct.pdf(x_cont0, loc=loc_in, scale=scale_in, nc=skew_in, df=tail1_in)*(bins[i+1]-bins[i])
           y_cont  = nct.pdf( x_cont,  loc=loc_in, scale=scale_in, nc=skew_in, df=tail1_in)*(bins[i+1]-bins[i])
           mycolor = "lime"#'#00ced1'#
           ax.plot(x_cont0, y_cont0, color='black', lw=5.2)
           if not (plot_several_curves): ax.plot(x_cont, y_cont, '.', color='black', markersize=17)
           ax.plot(x_cont0, y_cont0, label='Fit to t-student (nct)', color=mycolor, lw=3)
           pathout = f"{fitted_time_series.directory_output_plots}/histofit_{name_plot}_nct.pdf"
        elif ( fitting_function =="levy_stable"):
           y_cont0 = levy_stable.pdf( x_cont0 , loc=loc_in, scale=scale_in,  beta=skew_in, alpha=tail1_in )*(bins[i+1]-bins[i])
           y_cont  = levy_stable.pdf( x_cont,   loc=loc_in, scale=scale_in,  beta=skew_in, alpha=tail1_in)*(bins[i+1]-bins[i])
           mycolor = '#ffa500'#'#daa520'
           ax.plot(x_cont0, y_cont0, color='black', lw=5.2)
           if not (plot_several_curves): ax.plot(x_cont, y_cont, '.', color='black', markersize=17)
           ax.plot(x_cont0, y_cont0, label='Fit to stable', color=mycolor, lw=3)
           pathout = f"{fitted_time_series.directory_output_plots}/histofit_{name_plot}_stable.pdf"
        elif ( fitting_function=="genhyperbolic"):
           y_cont0 = genhyperbolic.pdf(x_cont0, loc=loc_in, scale=scale_in,  b=skew_in, a=tail1_in, p=tail2_in )*(bins[i+1]-bins[i])
           y_cont  = genhyperbolic.pdf( x_cont, loc=loc_in, scale=scale_in,  b=skew_in, a=tail1_in, p=tail2_in )*(bins[i+1]-bins[i])
           mycolor = "#ff1493"#"#FF00FF"
           ax.plot(x_cont0, y_cont0, color='black', lw=5.2)
           if not (plot_several_curves): ax.plot(x_cont, y_cont, '.', color='black', markersize=17)
           if ( fitted_time_series.consider_p):
               ax.plot(x_cont0, y_cont0, label='Fit to hyperbolic (with p)', color=mycolor, lw=3)
               pathout = f"{fitted_time_series.directory_output_plots}/histofit_{name_plot}_hyperbolicwithp.pdf"
           else:
               ax.plot(x_cont0, y_cont0, label='Fit to hyperbolic', color=mycolor, lw=3)
               pathout = f"{fitted_time_series.directory_output_plots}/histofit_{name_plot}_hyperbolicwop.pdf"
        if not (plot_several_curves):
            ax.plot(x_cont, y_cont, '.', label='Fit to ' + str( fitting_function),  color=mycolor,  markersize=14)

        del x_cont; del x_cont0; del y_cont0; del y_cont; del scale_in; del skew_in; del tail1_in; del tail2_in; del mycolor;
    '''

    pathout = '/Users/pgr/PyCharmProjects/PaperStopStrategy/Output/histogram_'+name.replace(" ","_")+'.pdf'
    pd.DataFrame(array_in,columns=["price"]).to_csv(pathout.replace(".pdf",".csv"),index=False)

    # Add labels to the plot:
    ax.set_ylabel('Probability',fontsize=15)
    ax.set_xlabel('Spread residual',fontsize=14)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    mytitle = str(name)
    mytitle = mytitle.replace("Yield","Price"); mytitle = mytitle.replace("YIELD","Price"); mytitle = mytitle.replace("yield","Price"); # We change this because in the class definition we have calculated returns from prices, not from yields.
    #mytitle = rewrite_title( mytitle )
    ax.set_title( mytitle ,fontsize=16 )
    ax.set_xticks(x)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    plt.xticks(rotation=45, ha='right')
    #plt.legend(loc='upper right')
    order = [1, 0]
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 11})
    plt.ylim([0.0, 0.35])
    #ax.set_xticklabels(labels)
    handles, labels = plt.gca().get_legend_handles_labels()
    ''' 
    if (loc_in != None):
        if not (plot_several_curves):
            order = [2, 0]
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 11})
        else:
            order = [4,0,1,2,3]
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 11}, ncol=2, labelspacing=0.4)
    '''
    #plt.legend(loc='upper right' )
    fig.tight_layout()

    #del len_to_remove; del lim_bins; del num_bins; del labels; del x; del width; del counts; del counts0

    #plt.show()

    # Save the plot:
    plt.savefig( pathout, format="pdf", bbox_inches="tight")
    plt.clf(); plt.cla(); plt.close('all')
    del ax; del fig

    return

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
