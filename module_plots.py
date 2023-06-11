from itertools import product
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
#from scipy.interpolate import make_interp_spline
from os import makedirs, path
from scipy.stats import norm, nct, genhyperbolic, levy_stable
from scipy.interpolate import make_interp_spline
from module_parameters import plot_labels, list_columns, distribution_types_text3
#pd.set_option('max_columns', 20); pd.set_option('max_rows', 99)

#-----------------------------------------------------------------------------------------------


# ===============================================================================================================
#             BLOCK FOR PLOTS CORRESPONDING TO FITTING
# ===============================================================================================================

#-----------------------------------------------------------------------------------------------------------------------

def plot_autocorrelation( ts, filepath_plot, my_title ):

    #from tsaplots import plot_acf
    from statsmodels.graphics.tsaplots import plot_acf

    plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
    # plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    fig, axes = plt.subplots(1, 1)
    fig = plot_acf( ts, lags=8, ax=axes)
    axes.set_title(my_title)
    axes.set_xlabel(r'Lag', fontsize=16)
    axes.tick_params(axis="x", labelsize=14);
    axes.tick_params(axis="y", labelsize=14)  # ; axes.set_yticks(fontsize=14)
    axes.set_ylabel(r'Autocorrelation of residuals', fontsize=16)
    #plt.savefig("autocorrelation_dW.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(filepath_plot, format="pdf", bbox_inches="tight")
    plt.clf(); plt.cla(); plt.close('all')

    del axes; del fig

    return

#-----------------------------------------------------------------------------------------------------------------------

def plot_2D( datafile_path, output_dir, plot_name, quantity_x, quantity_y, quantity_y2=None, rescaling_y2=1, mytitle=None  ):
    '''This function makes a 2D plot whose variable x is "quantity_x" and whos variable y is the optimum (maximum or
     minimum of "quantity_y").'''

    list_product_labels_name = output_dir.replace("Time_series/Spreads/Spreads_","");  list_product_labels_name = list_product_labels_name.replace("/Plots","")
    if ( list_product_labels_name=="cryptocurrencies" ): my_interval=91#365
    else: my_interval =252

    lt1 = plot_name.replace("spr_",""); lt2 = lt1.split("_")[1] ;  lt1 = lt1.split("_")[0]
    axes_text = {"Spread_y_vs_x":"Spread","Spread_x_vs_y":"Spread", "Sharpe_ratio":"$SR$", "Sharpe_ratio_with_semideviation":"$SR'$", "Sharpe_ratio_from_semideviation":"$SR'$", "profit_taking_param":"profit-taking ($)"}
    if (quantity_y=="Spread_y_vs_x"):
        legend_text = lt2+"-vs-"+lt1
    elif (quantity_y=="Spread_x_vs_y"):
        legend_text = lt1+"-vs-"+lt2
    else:
        try:
            legend_text = axes_text[quantity_y]
        except KeyError:
            legend_text = quantity_y

    # Reading the data
    df0 = pd.read_csv( datafile_path, header=0 , usecols=[quantity_x, quantity_y] )
    print(datafile_path,"\n",df0)
    arr_x = df0[quantity_x]; arr_y = df0[quantity_y]
    text=""

    # Actual plotting
    fig, ax = plt.subplots()
    try:
        label_x = axes_text[quantity_x]
    except KeyError:
        label_x = quantity_x
    plt.xlabel( label_x, fontsize=16)
    try:
        plt.ylabel( "Sharpe ratio", fontsize=16) #xxx plt.ylabel( axes_text[quantity_y], fontsize=16)
    except KeyError:
        plt.ylabel( quantity_y, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='both', which='minor', labelsize=11)
    plt.xticks(rotation=45, ha='right')

    if (quantity_x=="Date"):
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=my_interval))
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    color1 = 'b'
    color2 = '#8b0000'

    #ax.plot( arr_x, arr_y, '-', label=legend_text, color=color1, linewidth=1.2, markersize=2 )
    #ax.plot( [arr_x[0],arr_x[660]], [1,1], '-',   color='#ff4500', linewidth=3.5 )
    #ax.plot([arr_x[720], arr_x[2380]], [1.18, 1.18], '-',  color='#ff4500', linewidth=3.5)
    #ax.plot([arr_x[2510], arr_x[len(arr_x)-1]], [1.412, 1.412], '-',  color='#ff4500', linewidth=3.5)
    ax.plot(arr_x, arr_y, "o", label=legend_text, color=color1, linewidth=2, markersize=5)
    text = "-and-" + str(quantity_y2)
    try:
        x_smooth = np.linspace(arr_x.min(), arr_x.max(), 300)
        y_smooth = np.array(make_interp_spline(arr_x, arr_y)(x_smooth))
        ax.plot(x_smooth, y_smooth, '-', color=color1, linewidth=2.2)
        del x_smooth; del y_smooth
    except ValueError:
        print("\nWARNING: Unable to perform the interpolation of " + quantity_y2 + "-vs-" + quantity_x + "\n")


    if (quantity_y2!=None):
        df1 = pd.read_csv( datafile_path, header=0, usecols=[quantity_x, quantity_y2] )
        print(datafile_path,"\n",df1)
        arr_x2 = df1[quantity_x]; arr_y2 = df1[quantity_y2]
        arr_y2 *= rescaling_y2
        if (quantity_y2 == "Spread_y_vs_x"):
            legend_text2 = lt2 + "-vs-" + lt1
        elif (quantity_y2 == "Spread_x_vs_y"):
            legend_text2 = lt1 + "-vs-" + lt2
        else:
            try:
                legend_text2 = axes_text[quantity_y2]
            except KeyError:
                legend_text2 = quantity_y2
        #if (quantity_y2 == "Sharpe_ratio_with_semideviation"): my_label2 = ("Sharpe ratio from\nsemi-deviation "+r'($\times 1/\sqrt{2}$' + ")")
        ax.plot(arr_x2, arr_y2, "s", label=legend_text2, color=color2, linewidth=2, markersize=5)
        text = "-and-" + str(quantity_y2)
        try:
            x2_smooth = np.linspace(arr_x2.min(), arr_x2.max(), 300)
            y2_smooth = np.array( make_interp_spline(arr_x2, arr_y2)(x2_smooth) )
            ax.plot(x2_smooth, y2_smooth, '--', color=color2, linewidth=2)
            del x2_smooth; del y2_smooth
        except ValueError:
            print("\nWARNING: Unable to perform the interpolation of " + quantity_y2 + "-vs-" + quantity_x + "\n")

        del arr_x2; del arr_y2; del df1

    if (mytitle!=None):
        ax.set_title(mytitle,fontsize=16 )
    plt.legend(loc='best', labelspacing=1.5) # loc='lower right', loc='best', bbox_to_anchor=(0.5, 0.1, 0.5, 0.5),
    plot_path = output_dir+"/"+plot_name +"_"+quantity_y+".pdf"
    plt.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.clf(); plt.cla(); plt.close('all')

    del quantity_x; del quantity_y; del ax; del fig; del arr_y; del arr_x;

    return

#-----------------------------------------------------------------------------------------------

def make_all_plots_2D( input_params , quantity_x="Date", list_quantity_y=["Spread_y_vs_x","Spread_x_vs_y"], input_files_dir="Time_series/Spreads", input_files_prefix="spr_", plots_dir="Time_series/Plots" ):
    '''This function makes e.g. the plots of all the spreads.'''

    if not (input_params.make_plots): return

    df0 = pd.read_csv( input_params.file_corr_path, header=0, usecols=["product_1", "product_2"])

    if (input_files_dir=="Time_series/Spreads"):
        input_files_dir = "Time_series/Spreads/Spreads_"+input_params.list_product_labels_name + "/Data"
        plots_dir =       "Time_series/Spreads/Spreads_"+input_params.list_product_labels_name + "/Plots"

    if not ( path.exists(input_files_dir)): makedirs(input_files_dir)

    for quantity_y in list_quantity_y:
        for prod_label1, prod_label2 in zip(df0["product_1"], df0["product_2"]):
            plot_name = input_files_prefix+prod_label1 +"_"+ prod_label2
            plot_2D(input_files_dir+"/"+plot_name+".csv", plots_dir, plot_name, quantity_x, quantity_y )
        print("* The plots of ",quantity_y,"vs",quantity_x,"were performed. Find them at",plots_dir)

    del input_params; del quantity_x;del list_quantity_y; del quantity_y;del input_files_dir;del input_files_prefix;del plots_dir;del prod_label1;del prod_label2;del df0

    return

#-----------------------------------------------------------------------------------------------

def rewrite_title( mytitle ):
    mytitle = mytitle.replace(".csv","");
    string_aux_0 = mytitle.replace("spr_resid_","")
    string_aux1 = string_aux_0.split("_")[0]
    string_aux2 = string_aux_0.split("_")[1]
    mytitle = string_aux_0
    for stri in ["spr_resid_", string_aux1+"_"+string_aux2+"_", string_aux2+"_"+string_aux1+"_", "__"]: mytitle = mytitle.replace(stri,"")
    if (mytitle == "x_vs_y"):
        mytitle = "Residuals of spreads of " + str(string_aux2) + " / " + str(string_aux1) # string_aux_1 is x; string_aux_2 is y
    elif (mytitle=="y_vs_x"):
         mytitle = "Residuals of spreads of "+ str(string_aux1) +" / "+ str(string_aux2)
    else:
        print("WARNING: mytitle is ",mytitle," which is neither 'y_vs_x' nor 'x_vs_y'.")

    del string_aux1; del string_aux2; del string_aux_0; del stri
    return mytitle

#-----------------------------------------------------------------------------------------------

def provide_curve_parameters( fitting_function, fitted_time_series, plot_several_curves, params_several_curves ):
    skew_in = None;tail1_in = None;tail2_in = None
    if not (plot_several_curves):
        loc_in = fitted_time_series.fitting_parameters['loc_param']
        scale_in = fitted_time_series.fitting_parameters['scale_param']
        if (fitting_function == "nct"):
            skew_in = fitted_time_series.fitting_parameters['skewness_param']
            tail1_in = fitted_time_series.fitting_parameters['df_param']
            # tail2_in = fitted_time_series.fitting_parameters['distribution_type']
        elif (fitting_function == "levy_stable"):
            skew_in = fitted_time_series.fitting_parameters['beta_param']
            tail1_in = fitted_time_series.fitting_parameters['alpha_param']
        elif (fitting_function == "genhyperbolic"):
            skew_in = fitted_time_series.fitting_parameters['b_param']
            tail1_in = fitted_time_series.fitting_parameters['a_param']
            tail2_in = fitted_time_series.fitting_parameters['p_param']
    else:
        my_params =  params_several_curves[fitting_function]
        loc_in = my_params['loc_param']
        scale_in = my_params['scale_param']
        if (fitting_function == "nct"):
            skew_in = my_params['skewness_param']
            tail1_in = my_params['df_param']
        elif (fitting_function == "levy_stable"):
            skew_in = my_params['beta_param']
            tail1_in = my_params['alpha_param']
        elif (fitting_function == "genhyperbolic"):
            skew_in = my_params['b_param']
            tail1_in = my_params['a_param']
            tail2_in = my_params['p_param']
    return loc_in, scale_in, skew_in, tail1_in, tail2_in

# -----------------------------------------------------------------------------------------------

def plot_histogram(  fitted_time_series, plot_several_curves=False, params_several_curves=None ):
    ''' This function plots a histogram.
    '''

    if (plot_several_curves):
        list_sweep = ['norm', 'levy_stable', 'genhyperbolic', 'nct']
    else:
        list_sweep = [fitted_time_series.fitting_parameters['distribution_type']]

    # Define the bins
    len_to_remove = int(0.01 * len(fitted_time_series.ts_to_fit))
    lim_bins =  4*np.std( fitted_time_series.ts_to_fit[len_to_remove:-len_to_remove] )
    num_bins = 11
    bins = [-lim_bins  ] # OLD: [-lim_bins + loc_in ]; Si el loc se calculó mal, esto hace que el histograma tb se pinte mal.
    labels = []
    for i in range(1,num_bins+1):
        bins   += [ -lim_bins + i*(2*lim_bins)/num_bins ] #old:  [ -lim_bins + loc_in  + i*(2*lim_bins)/num_bins ]
        if (i%2 == 0):
          if (lim_bins >= 1):
            labels += [ "{:.4f}".format( -lim_bins  + (i-0.5)*(2*lim_bins)/num_bins  ) ] # old: labels += [ "{:.4f}".format( -lim_bins + loc_in  + (i-0.5)*(2*lim_bins)/num_bins  ) ]
          else:
            labels += ["{:.5f}".format(-lim_bins  + (i - 0.5) * (2 * lim_bins) / num_bins)] # old:  ["{:.5f}".format(-lim_bins + loc_in  + (i - 0.5) * (2 * lim_bins) / num_bins)]
        else:
            labels += [None]

    # Calculate values of the "float" histogram
    counts0, bins = np.histogram( fitted_time_series.ts_to_fit, bins )
    counts = np.zeros(len(counts0))
    for i in range(len(counts0)):
        counts[i] = counts0[i]/len(fitted_time_series.ts_to_fit)

    # Plot the histogram
    x = np.array( [ (bins[i]+bins[i+1])/2 for i in range(len(labels)) ] ) # old: np.arange(len(labels))

    width = 0.91*(bins[i+1]-bins[i])
    fig, ax = plt.subplots()
    ax.bar(x , counts, width, label='Histogram (normalized)', color="darkblue") #'mediumblue'

    # Plot fitting function:
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

    # Add labels to the plot:
    if not (plot_several_curves):
        plt.ylim([0,  max(counts) *1.3 ] )
    else:
        plt.ylim([0, max(counts) * 1.5])
    ax.set_ylabel('Probability',fontsize=15)
    ax.set_xlabel('Spread residual',fontsize=14)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    mytitle = str(fitted_time_series.filename)
    mytitle = mytitle.replace("Yield","Price"); mytitle = mytitle.replace("YIELD","Price"); mytitle = mytitle.replace("yield","Price"); # We change this because in the class definition we have calculated returns from prices, not from yields.
    mytitle = rewrite_title( mytitle )
    ax.set_title( mytitle ,fontsize=16 )
    ax.set_xticks(x)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    plt.xticks(rotation=45, ha='right')
    #ax.set_xticklabels(labels)
    handles, labels = plt.gca().get_legend_handles_labels()
    if (loc_in != None):
        if not (plot_several_curves):
            order = [2, 0]
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 11})
        else:
            order = [4,0,1,2,3]
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 11}, ncol=2, labelspacing=0.4)

    #plt.legend(loc='upper right' )
    fig.tight_layout()

    del len_to_remove; del lim_bins; del num_bins; del labels; del x; del width; del counts; del counts0

    #plt.show()

    # Save the plot:
    if (plot_several_curves): pathout=pathout.replace(".pdf","_all.pdf")
    plt.savefig( pathout, format="pdf", bbox_inches="tight")
    plt.clf(); plt.cla(); plt.close('all')
    del ax; del fig

    return

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_histograms_without_fitting(input_params):
    '''This function makes the histogram plots without actual fitting'''

    if not (input_params.make_plots): return

    from module_fitting import define_names, FittedTimeSeries

    print("input_params.file_stationarity_path",input_params.file_stationarity_path)

    df_stationary = pd.read_csv(input_params.file_stationarity_path,header=0)
    df_stationary = df_stationary[ df_stationary['probably_stationary']==True ]
    suffix = { 'normal':'norm','norm':'normal','levy_stable':'stable','nct':'nct','genhyperbolic':'genhyperbolic_withp' }
    cols = {'norm':'normal','genhyperbolic':'ghyp', 'nct':'nct', 'levy_stable':'stable'}

    for i in df_stationary.index:
        filename_resid, name_col_to_fit, trash = define_names(input_params, df_stationary.loc[i,"products"], df_stationary.loc[i,"quantity"])
        mydataset = FittedTimeSeries( input_params, filename_resid, name_col_to_fit )
        for distrib_type in input_params.list_distribution_types:
            filepath = input_params.ts_directory + "/Spreads/Spreads_" + input_params.list_product_labels_name + "/Data/Fitting_parameters/spr_fitting_params_" + input_params.list_product_labels_name + "_norm_nct_genhyperbolic_levy_stable.csv"
            if not (path.exists(filepath)):
                filepath = input_params.ts_directory + "/Spreads/Spreads_" + input_params.list_product_labels_name + "/Data/Fitting_parameters/spr_fitting_params_" + input_params.list_product_labels_name +"_"+ suffix[distrib_type] +".csv"
            if not (path.exists(filepath)):
                if (suffix[distrib_type]=='genhyperbolic_withp'):
                    filepath = filepath.replace("genhyperbolic_withp","genhyperbolic")
                    if not (path.exists(filepath)):
                        print("\nWARNING: The file", filepath, "does not exit. Omitting.")
                        continue
                else:
                    print("\nWARNING: The file",filepath,"does not exit. Omitting.")
                    continue

            df_fitting_params = pd.read_csv(filepath,header=0)
            df_fitting_params.set_index("spread_name", inplace=True)
            fitting_params_one = df_fitting_params.loc[filename_resid]  # spr_resid_MPC_TTE_y_vs_x.csv

            if (distrib_type=='norm'):
                mydataset.fitting_parameters = {'distribution_type': distrib_type,'loc_param': fitting_params_one[cols[distrib_type] + "_loc"],'scale_param': fitting_params_one[cols[distrib_type] + "_scale"],'loss': fitting_params_one[cols[distrib_type] + "_loss"]}
            elif (distrib_type=='nct'):
                mydataset.fitting_parameters = {'distribution_type': distrib_type, 'loc_param': fitting_params_one[cols[distrib_type]+"_loc"],
                                            'scale_param': fitting_params_one[cols[distrib_type]+"_scale"], 'skewness_param': fitting_params_one[cols[distrib_type]+"_skparam"],
                                            'df_param': fitting_params_one[cols[distrib_type]+"_dfparam"], 'loss': fitting_params_one[cols[distrib_type]+"_loss"]}
            elif (distrib_type == 'genhyperbolic'):
                mydataset.fitting_parameters = {'distribution_type': distrib_type,
                                                'loc_param': fitting_params_one[cols[distrib_type] + "_loc"],
                                                'scale_param': fitting_params_one[cols[distrib_type]+ "_scale"],
                                                'b_param': fitting_params_one[cols[distrib_type]+ "_b_param"],
                                                'a_param': fitting_params_one[cols[distrib_type] + "_a_param"],
                                                'p_param': fitting_params_one[cols[distrib_type] + "_p_param"],
                                                'loss': fitting_params_one[cols[distrib_type]+ "_loss"]}
            elif (distrib_type == 'levy_stable'):
                mydataset.fitting_parameters = {'distribution_type': distrib_type,
                                                'loc_param': fitting_params_one[cols[distrib_type] + "_loc"],
                                                'scale_param': fitting_params_one[cols[distrib_type] + "_scale"],
                                                'beta_param': fitting_params_one[cols[distrib_type] + "_beta_param"],
                                                'alpha_param': fitting_params_one[cols[distrib_type] + "_alpha_param"],
                                                'loss': fitting_params_one[cols[distrib_type] + "_loss"]}
            mydataset.plot_fitting()
            del fitting_params_one

    del df_stationary;  del df_fitting_params; del input_params; del mydataset; del filename_resid; del name_col_to_fit

    return

# ------------------------------------------------------------------------------------------------

def plot_histograms_without_fitting_all_curves(input_params):
    '''This function makes the histogram plots without actual fitting'''

    from module_fitting import define_names, FittedTimeSeries

    if not (input_params.make_plots): return

    list_to_sweep = ['norm','nct','genhyperbolic','levy_stable']

    df_stationary = pd.read_csv(input_params.file_stationarity_path,header=0)
    df_stationary = df_stationary[ df_stationary['probably_stationary']==True ]
    suffix = { 'normal':'norm','norm':'norm','levy_stable':'stable','nct':'nct','genhyperbolic':'genhyperbolic_withp' }
    cols = {'norm':'normal','genhyperbolic':'ghyp', 'nct':'nct', 'levy_stable':'stable'}

    for i in df_stationary.index:
        filename_resid, name_col_to_fit, trash = define_names(input_params, df_stationary.loc[i,"products"], df_stationary.loc[i,"quantity"])
        mydataset = FittedTimeSeries( input_params, filename_resid, name_col_to_fit )
        all_params = {'norm':None,'levy_stable':None,'nct':None,'genhyperbolic':None}
        for distrib_type in list_to_sweep:

            filepath = input_params.ts_directory + "/Spreads/Spreads_" + input_params.list_product_labels_name + "/Data/Fitting_parameters/spr_fitting_params_" + input_params.list_product_labels_name + "_norm_nct_genhyperbolic_levy_stable.csv"
            if not (path.exists(filepath)):
                filepath = input_params.ts_directory + "/Spreads/Spreads_" + input_params.list_product_labels_name + "/Data/Fitting_parameters/spr_fitting_params_" + input_params.list_product_labels_name +"_"+ suffix[distrib_type] +".csv"
            if not (path.exists(filepath)):
                print("\n WARNING: file",filepath,"not found. Omitting the corresponding plots.\n")
                continue
            df_fitting_params = pd.read_csv(filepath,header=0)
            df_fitting_params.set_index("spread_name", inplace=True)

            try:
                fitting_params_one = df_fitting_params.loc[filename_resid]  # spr_resid_MPC_TTE_y_vs_x.csv
            except KeyError:
                continue
            if (distrib_type=='norm'):
                mydataset.fitting_parameters = {'distribution_type': distrib_type,'loc_param': fitting_params_one[cols[distrib_type] + "_loc"],'scale_param': fitting_params_one[cols[distrib_type] + "_scale"],'loss': fitting_params_one[cols[distrib_type] + "_loss"]}
            elif (distrib_type=='nct'):
                mydataset.fitting_parameters = {'distribution_type': distrib_type, 'loc_param': fitting_params_one[cols[distrib_type]+"_loc"],
                                            'scale_param': fitting_params_one[cols[distrib_type]+"_scale"], 'skewness_param': fitting_params_one[cols[distrib_type]+"_skparam"],
                                            'df_param': fitting_params_one[cols[distrib_type]+"_dfparam"], 'loss': fitting_params_one[cols[distrib_type]+"_loss"]}
            elif (distrib_type == 'genhyperbolic'):
                mydataset.fitting_parameters = {'distribution_type': distrib_type,
                                                'loc_param': fitting_params_one[cols[distrib_type] + "_loc"],
                                                'scale_param': fitting_params_one[cols[distrib_type]+ "_scale"],
                                                'b_param': fitting_params_one[cols[distrib_type]+ "_b_param"],
                                                'a_param': fitting_params_one[cols[distrib_type] + "_a_param"],
                                                'p_param': fitting_params_one[cols[distrib_type] + "_p_param"],
                                                'loss': fitting_params_one[cols[distrib_type]+ "_loss"]}
            elif (distrib_type == 'levy_stable'):
                mydataset.fitting_parameters = {'distribution_type': distrib_type,
                                                'loc_param': fitting_params_one[cols[distrib_type] + "_loc"],
                                                'scale_param': fitting_params_one[cols[distrib_type] + "_scale"],
                                                'beta_param': fitting_params_one[cols[distrib_type] + "_beta_param"],
                                                'alpha_param': fitting_params_one[cols[distrib_type] + "_alpha_param"],
                                                'loss': fitting_params_one[cols[distrib_type] + "_loss"]}
            all_params[distrib_type] = mydataset.fitting_parameters.copy()

        mydataset.plot_fitting(True,all_params)


    del df_stationary; del df_fitting_params; del fitting_params_one; del input_params; del mydataset; del filename_resid; del name_col_to_fit

    return

# ------------------------------------------------------------------------------------------------


# ===============================================================================================================
#             BLOCK FOR PLOTS CORRESPONDING TO TRADING RULES
# ===============================================================================================================


# pd.set_option('max_columns', 20); pd.set_option('max_rows', 99999)

# =======================================================================================================================
#                                     BLOCK FOR STORING DATA
# =======================================================================================================================

# -----------------------------------------------------------------------------------------------

def reformat_param(distrib_param):
    '''This function rewrites an input number to the appropriate format'''
    if distrib_param == None:
        return None
    if (isinstance(distrib_param, float)):
        return str("{:.4f}".format(distrib_param))
    if (isinstance(distrib_param, int)):
        return str(distrib_param)
    return distrib_param


# ---------------------------------------------------------------------------------------------------------------------

def define_results_name(input_params, OU_params, rv_params):
    '''This function prints the results to screen and stores them in a .csv file.'''

    distr = rv_params['distribution_type']

    if (input_params.path_rv_params != None):

        res_name = "OrnUhl-" + (OU_params["product_label"]).replace(".csv", "") + "-" + distr

    else:  # rv parameters read from input.py, not from a fitting .csv file

        if (input_params.evolution_type == "Ornstein-Uhlenbeck_equation"):
            E0 = OU_params['E0']
            tau = OU_params['tau'];
            if (isinstance(E0, float)):   E0 = "{:.4f}".format(E0)
            if (isinstance(tau, float)):  tau = "{:.4f}".format(tau)
            res_name = "OrnUhl-E0_" + str(E0) + "__tau_" + str(tau) + "-" + distr + "_"
        else:
            res_name = "single-" + distr + "_"

        mu = reformat_param(rv_params['mu'])
        sigma = reformat_param(rv_params['sigma'])
        third_param = reformat_param(rv_params['third_param'])  # Skewness parameter in t-student
        fourth_param = reformat_param(rv_params['fourth_param'])  # Degrees of freedom parameter in t-student
        fifth_param = reformat_param(rv_params['fifth_param'])

        if ( distr == 'norm'):
            res_name += "mu_" + mu + "__sigma_" + sigma
        elif ( distr == 'nct'):
            res_name += "mu_" + mu + "__sigma_" + sigma + "__sk_" + third_param + "__df_" + fourth_param
        elif ( distr == 'genhyperbolic'):
            res_name += "mu_" + mu + "__sigma_" + sigma + "__3p_" + third_param + "__4p_" + fourth_param + "__5p_" + fifth_param
        elif ( distr == 'levy_stable'):
            res_name += "mu_" + mu + "__sigma_" + sigma + "__3p_" + third_param + "__4p_" + fourth_param

    if ( input_params.poisson_probability != None ):
        if ( abs(input_params.poisson_probability) < 0.0000000001 ):
            res_name += "_Poisson"

    del input_params; del OU_params; del rv_params

    return res_name

# -----------------------------------------------------------------------------------------

def save_and_plot_results(input_params, rv_params, df_out, OU_params=None, makeplo=True ):
    '''This function stores the dataframe which contains the results of a heatmap (if the dataframe is not empty)
    and makes the corresponding plots.'''

    datafile_name, plot_name = print_and_save_results(input_params, OU_params, rv_params, df_out)

    if not ( (input_params.make_plots==True) and (makeplo==True) ):
        del input_params; del rv_params; del df_out; del OU_params; del makeplo
        return

    plot_heatmaps(input_params, datafile_name, plot_name, rv_params['distribution_type'])

    if (input_params.evolution_type == "Ornstein-Uhlenbeck_equation"): x_2D_plots = "enter_value"
    else: x_2D_plots = "profit_taking_param"
    for quantity in ["ES", "Sharpe_ratio", "Sharpe_ratio_with_semideviation"]: # profit_mean, profit_std and semideviation are typically monotonic
        plot_optima_2D(datafile_name, input_params.output_trad_rules_dir, plot_name, x_2D_plots, quantity)
    plot_optima_2D(datafile_name, input_params.output_trad_rules_dir, plot_name, x_2D_plots, "Sharpe_ratio","Sharpe_ratio_with_semideviation", 1 / np.sqrt(2))

    del input_params; del rv_params; del df_out; del OU_params

    return

# ----------------------------------------------------------------------------------------

def print_and_save_results(input_params, OU_params, rv_params, df_out):
    '''This function prints the results to screen and stores them in a .csv file.'''

    distr = rv_params['distribution_type']

    if (input_params.evolution_type == "Ornstein-Uhlenbeck_equation"):
        E0 = OU_params['E0']
        tau = OU_params['tau'];
        if (isinstance(E0, float)):   E0 = "{:.4f}".format(E0)
        if (isinstance(tau, float)):  tau = "{:.4f}".format(tau)

    mu = reformat_param(rv_params['mu'])
    sigma = reformat_param(rv_params['sigma'])
    third_param = reformat_param(rv_params['third_param'])  # Skewness parameter in t-student
    fourth_param = reformat_param(rv_params['fourth_param'])  # Degrees of freedom parameter in t-student
    fifth_param = reformat_param(rv_params['fifth_param'])

    if not (input_params.only_plots):
        print("\n\n----------------------------------------------------------------------------")
        if (input_params.path_rv_params != None): print("               ",OU_params['product_label'])
        if (input_params.evolution_type  == "Ornstein-Uhlenbeck_equation"):

            print("          Ornstein-Uhlenbeck equation with E0 =", E0, ", tau =", tau)
            rv_kind = "residuals"
            text1 = "and "
        elif (input_params.evolution_type  == "single_product"):
            print("         Single product")
            rv_kind = "random variable"
            text1 = ""
        if ( distr == "norm"):
            print("          " + text1 + "Gaussian", rv_kind, "with loc =", mu, ", scale =", sigma)
        elif ( distr == "nct"):
            print("           " + text1 + "t-student distributed", rv_kind, "with\n loc =", mu, ", scale =", sigma,", skewness_param=", third_param, ", df_param=", fourth_param)
        elif ( distr == "genhyperbolic"):
            print("           " + text1 + "hyperbolically distributed", rv_kind, "with\n loc =", mu, ", scale =", sigma,", skewness (b) param=", third_param, ", tail (a) param=", fourth_param, ", p param=", fifth_param)
        elif (distr == "levy_stable"):
            print("           " + text1 + "levy stable distributed", rv_kind, "with\n loc =", mu, ", scale =", sigma,", skewness (beta) param=", third_param, ", tail (alpha) param=", fourth_param )
        else:
            raise Exception("\nERROR: Unknown distribution "+distr+".\n")

        print("----------------------------------------------------------------------------")

    res_name = define_results_name(input_params, OU_params, rv_params)
    datafile_name = f'{input_params.output_trad_rules_dir}/Results/{res_name}.csv'

    if ((not df_out.empty) and (not (input_params.only_plots)) ):
        print(df_out)
        df_out.to_csv(datafile_name, index=True)

    del df_out; del OU_params; del input_params; del rv_params

    return datafile_name, res_name


# ----------------------------------------------------------------------------------------


# =======================================================================================================================
#                                     BLOCK FOR PLOTTING
# =======================================================================================================================

def convert_df_for_heatmap(df_in, x_axis_name, y_axis_name, repr_column_name="Sharpe_ratio"):
    '''This function converts an input dataframe with a double index and one data column to a "chessboard" format
    which can be easily used to plot a heatmap by the seaborn module'''

    if (df_in.empty): return None, None, None, repr_column_name

    if (repr_column_name=="Sharpe_ratio_with_semideviation"):
        rescaling = 1/np.sqrt(2)
        print("   WARNING: The heatmap of the Sharpe ratio from semideviation is rescaled with a 1/sqrt(2) multiplicative term.\n")
    else:
        rescaling = 1

    df_in = pd.DataFrame(df_in[repr_column_name])

    df1 = df_in.copy().reset_index()
    list_x = df1[x_axis_name].drop_duplicates().values.tolist()
    list_y = df1[y_axis_name].drop_duplicates().values.tolist()

    if (y_axis_name in ["max_horizon", "profit_taking_param"]): list_y.sort(reverse=True)

    data_in_chessboard = []
    for y in list_y:
        li_aux = []
        for x in list_x:
            li_aux.append( rescaling * (df_in.loc[(x, y), repr_column_name]) )
        data_in_chessboard.append(li_aux)

    del df1; del li_aux

    return data_in_chessboard, list_x, list_y, repr_column_name


# ----------------------------------------------------------------------------------------

def plot_heatmaps_pt_vs_en( input_params, list_quant_plot=["Sharpe_ratio","Sharpe_ratio_with_semideviation","profit_mean","profit_std","semideviation","VaR","ES","probab_loss"] ):
    '''This function makes plots of heatmaps of profit-taking vs enter-value for the spreads whose stationarity is True
    in the corresponding stationarity file, and whose fitting parameters are known (stored in files).'''

    if not (input_params.make_plots): return

    print(" * Now plotting heatmaps of profit-taking vs enter-value.")

    horiz_text = { 21:"one month", 42:"two months", 63:"one quarter",84:"four months", 126:"one semester", 252:"one year" }

    df_stationary = pd.read_csv(input_params.file_stationarity_path, header=0)
    df_stationary = df_stationary[df_stationary['probably_stationary'] == True]
    for ind in df_stationary.index:
        prod = df_stationary.loc[ind,"products"]
        prod = prod.replace("-vs-", "_")
        quant = df_stationary.loc[ind, "quantity"]
        quant = quant.replace("Spread_","")
        if (quant=="y_vs_x"):
            aux1 = prod.split("_")[0]
            aux2 = prod.split("_")[1]
            prod = aux2 + "_" + aux1
            del aux1; del aux2
        filecode = prod + "_" + quant

        for distrib in ['norm', 'nct', 'genhyperbolic', 'levy_stable']:
            filepath = 'Output/Output_trading_rules/Results/OrnUhl-spr_resid_'+filecode+'-'+distrib+'.csv'
            if not (path.exists(filepath)):
                continue
            df_results = pd.read_csv(filepath)
            if ( len( pd.DataFrame(df_results["enter_value"]).drop_duplicates() ) == 1 ): continue

            for col_name in ["enter_value","profit_taking_param"]:
                for ind in df_results.index:
                    if (abs(df_results.loc[ind, col_name]) < 0.0000000000001):
                        df_results.loc[ind, col_name] = 0

            li_horizon = ((df_results["max_horizon"]).drop_duplicates()).values.tolist()
            for max_horizon in li_horizon:
                for quant_to_plot in list_quant_plot:
                   if (max_horizon in horiz_text.keys()): horiz_title = " distrib.; horiz. " + horiz_text[max_horizon]
                   else: horiz_title = " distrib.; horiz. "+str(max_horizon) + " days"
                   sign_en = np.sign( np.mean( np.array( df_results["enter_value"] ) ) )
                   #if (sign_en > 0): sign_en = "positive_ev__"
                   #else: sign_en = "negative_ev__" #+sign_en+"horiz_"+ str(int(max_horizon)) + "__"
                   plot_one_heatmap(df_results,input_params.output_trad_rules_dir + '/Plots/hm_OrnUhl-spr_resid_'+filecode+"--"+ distrib + '.pdf', quant_to_plot, "max_horizon", max_horizon, "stop_loss_param", None, "enter_value","profit_taking_param","\n"+distribution_types_text3[distrib]+horiz_title )
            del df_results;

    del input_params; del df_stationary; del prod; del quant; del filecode; del distrib; del filepath;

    return

# ----------------------------------------------------------------------------------------

def plot_one_heatmap(df_results, plot_path, quantity_to_plot, quant_fixed1, value_fixed1, quant_fixed2, value_fixed2, quant_sweep_x, quant_sweep_y, subtitle=None):
    '''This function plots a heat-map and saves it.'''

    delt = 0.0000000001
    df1 = df_results.copy()

    #if ("norm.pdf" in plot_path): subtitle = ";  Normal distribution\nWith Poisson events"
    #elif ("nct.pdf" in plot_path): subtitle = ";  t-student distribution\nWith Poisson events"
    #elif ("genhyperbolic.pdf" in plot_path):subtitle = ";  gen. hyperbolic distribution\nWith Poisson events"
    #elif ("stable.pdf" in plot_path): subtitle = ";  stable distribution\nWith Poisson events"

    #print(df_results,"\nFIXED",quant_fixed1, value_fixed1, quant_fixed2, value_fixed2)

    if quant_fixed1 in ["profit_taking_param","stop_loss_param"]:
        df1[quant_fixed1] = df1[quant_fixed1] - df1["enter_value"]
    if quant_fixed2 in ["profit_taking_param","stop_loss_param"]:
        df1[quant_fixed2] = df1[quant_fixed2] - df1["enter_value"]

    if (value_fixed1 != None):
        df1 = df1[(df1[quant_fixed1] < value_fixed1 + delt) & (df1[quant_fixed1] > value_fixed1 - delt)]
        df1 = df1.drop(quant_fixed1, axis=1)


    if (value_fixed2 != None):
        df1 = df1[(df1[quant_fixed2] < value_fixed2 + delt) & (df1[quant_fixed2] > value_fixed2 - delt)]
        df1 = df1.drop(quant_fixed2, axis=1)

    if (((quant_sweep_x=="enter_value") and (quant_sweep_y=="profit_taking_param") ) or ((quant_sweep_y=="enter_value")  and (quant_sweep_x=="profit_taking_param") )):
        df1["profit_taking_param"] = df1["profit_taking_param"] - df1["enter_value"] # We express the profit-taking as a difference with enter-value because otherwise the heatmap cannot be plotted (without this modification each enter-value has a different range of profit-taking values).
        for ind in df1.index:
            df1.loc[ind,"profit_taking_param"] = "%.9f" % df1.loc[ind,"profit_taking_param"]

    # df1.to_csv(plot_path.replace(".pdf","_.csv"))

    if (df1.empty):
        del df_results; del plot_path; del quantity_to_plot
        return

    for col_name in [quant_sweep_x, quant_sweep_y]:
        for ind in df1.index:
            try:
                if (isinstance(df1.loc[ind,col_name], str)):
                    df1.loc[ind,col_name] = np.float( df1.loc[ind,col_name])
                if (abs( np.float( df1.loc[ind,col_name]) )<0.0000000000001):
                    df1.loc[ind, col_name] = 0
            except TypeError:
                raise Exception("\nERROR: Problem with the indices; index="+str(ind)+"colname="+col_name+"df="+str(df1.loc[ind,col_name])+"\n")

    df1 = df1.set_index([quant_sweep_x, quant_sweep_y])
    data_to_plot, li_x, li_y, plotted_quantity = convert_df_for_heatmap(df1, quant_sweep_x, quant_sweep_y,quantity_to_plot)

    if (len(li_x) > 1):
        periodicity_x = round(max(((li_x[-1] - li_x[0]) / (5 * (li_x[1] - li_x[0])), 2)))
    else:
        periodicity_x = 1
    if (len(li_y) > 1):
        periodicity_y = round(max(((np.float(li_y[-1]) - np.float(li_y[0])) / (5 * ( np.float(li_y[1]) - np.float(li_y[0]) ))), 2))
    else:
        periodicity_y = 1
    li_x1 = ["{:.3f}".format(li_x[i]) if ((i % periodicity_x) == 0) else None for i in range(len(li_x))]
    if (quant_sweep_y == "max_horizon"):
        li_y1 = ["{:.0f}".format(li_y[i]) if ((i % periodicity_y) == 0) else None for i in range(len(li_y))]
    else:
        li_y1 = ["{:.3f}".format(np.float(li_y[i]) ) if ((i % periodicity_y) == 0) else None for i in range(len(li_y))]

    ax = sns.heatmap(data_to_plot, linewidth=0.5, xticklabels=li_x1, yticklabels=li_y1, square=True, annot=False)

    l, b, w, h = ax.get_position().bounds
    ax.set_position([l * 1.125, b * 1.14, w * 0.94, h * 0.94])  # To change location and size of the figure;  [left, bottom, width, height]

    ax.set(xlabel=plot_labels[quant_sweep_x], ylabel=plot_labels[quant_sweep_y])
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')

    if (subtitle==None): title_suffix=""
    else: title_suffix=str(subtitle)
    ax.set_title( plot_labels[quantity_to_plot]+title_suffix, fontsize=14)  # fig.suptitle('test title', fontsize=20)

    # ax.xaxis.set_tick_params(length=5)
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, 1.3 * chartBox.y0, chartBox.width, chartBox.height])

    plot_path = plot_path.replace("OrnUhl-E0_", "HM_OrnUhl-E0_")  # HM stands for "heatmap"

    text = ""
    if (value_fixed1 != None):
        if (quant_fixed1=="max_horizon"): text += "__" + quant_fixed1 + "_" + "{:.0f}".format(value_fixed1)
        else: text += "__" + quant_fixed1 + "_" + "{:.3f}".format(value_fixed1)
    if (value_fixed2 != None):
        if (quant_fixed2=="max_horizon"): text += "__" + quant_fixed2 + "_" + "{:.0f}".format(value_fixed2)
        else: text += "__" + quant_fixed2 + "_" + "{:.3f}".format(value_fixed2)

    plot_path = plot_path.replace(".pdf", text + "-" + str(plotted_quantity) + ".pdf")
    plt.savefig(plot_path)
    # if (quantity_to_plot=="Sharpe_ratio"): plt.show()
    plt.clf()

    del ax;del text;del data_to_plot; del li_x;del li_x1;del li_y;del li_y1;del plotted_quantity;del plot_path; del df1;del df_results

    return


# ----------------------------------------------------------------------------------------

def plot_heatmaps(input_params, datafile_path, plot_name, distrib_type):
    '''This function plots a heat-map and saves it.'''

    if not (input_params.make_plots): return

    if (input_params.plot_heatmaps_for_individual_enter_value):
        print(" Now plotting heatmaps using data from "+datafile_path+"\n to"+input_params.output_trad_rules_dir+ '/Plots/')

    if not (path.exists(datafile_path)):
        print("\nWARNING: The file "+datafile_path+" does not exist; plots based on it cannot be plotted.\n")
        return
    df_results = pd.read_csv(datafile_path, header=0)

    for col_name in ["enter_value","profit_taking_param"]:
        for ind in df_results.index:
            if (abs(df_results.loc[ind,col_name])<0.0000000000001):
                df_results.loc[ind, col_name] = 0

    li_mh = df_results.copy();
    li_mh = li_mh["max_horizon"].drop_duplicates().values.tolist()
    if ((input_params.method_for_calculation_of_profits != "enter_ensured")):
        li_enter = df_results.copy();
        li_enter = li_enter["enter_value"].drop_duplicates().values.tolist()
    li_sl = input_params.list_stop_loss  # df_results.copy(); li_sl = li_sl["stop_loss_param"].drop_duplicates().values.tolist()

    for quantity_to_plot in list_columns:

        # Plot for many horizons
        if (len(pd.DataFrame(df_results["max_horizon"]).drop_duplicates())>1):
            plot_one_heatmap(df_results, input_params.output_trad_rules_dir + '/Plots/Many_horizons_' + plot_name + '.pdf',quantity_to_plot, "profit_taking_param", input_params.list_profit_taking[2], "stop_loss_param", min(li_sl),"enter_value", "max_horizon", "\n"+distribution_types_text3[distrib_type]+" distribution; \nSeveral horizons")

        if (input_params.evolution_type == "Ornstein-Uhlenbeck_equation"):

            if (input_params.plot_heatmaps_for_individual_enter_value):

                for en in li_enter:
                    plot_one_heatmap(df_results, input_params.output_trad_rules_dir + '/Plots/' + plot_name + '.pdf', quantity_to_plot, "enter_value", en, "stop_loss_param", en + min(li_sl),"profit_taking_param", "max_horizon")

                if ((input_params.method_for_calculation_of_profits != "enter_ensured")):
                    for mh in li_mh:
                        for en in li_enter:
                            plot_one_heatmap(df_results, input_params.output_trad_rules_dir + '/Plots/' + plot_name + '.pdf',quantity_to_plot, "max_horizon", mh, "enter_value", en, "profit_taking_param","stop_loss_param")

                '''  This heatmap would demand subtracting the enter_value to each profit_taking_param, otherwise the pt's vary for different en's.
                for mh in li_mh:
                    plot_one_heatmap(df_results, input_params.output_trad_rules_dir + '/Plots/' + plot_name + '.pdf',
                                     quantity_to_plot, "max_horizon", mh, "stop_loss_param", li_sl[0], "profit_taking_param","enter_value")
                '''

        else:  # input_params.evolution_type == "single_product"

            plot_one_heatmap(df_results, input_params.output_trad_rules_dir + '/Plots/' + plot_name + '.pdf',quantity_to_plot, "enter_value", None, "stop_loss_param", 1 + min(li_sl), "profit_taking_param", "max_horizon")

            for mh in li_mh:
                plot_one_heatmap(df_results, input_params.output_trad_rules_dir+ '/Plots/' + plot_name + '.pdf',quantity_to_plot, "max_horizon", mh, "enter_value", None, "profit_taking_param", "stop_loss_param")

    del datafile_path; del input_params; del plot_name; del quantity_to_plot; del df_results

    return

# ---------------------------------------------------------------------------------------

def read_data(datafile_path, quantity_x, quantity_y):
    if not (path.exists(datafile_path)): return None, None, False
    df0 = pd.read_csv(datafile_path, header=0, usecols=[quantity_x, quantity_y])
    arr_x = np.sort(df0[quantity_x].unique())
    df0 = df0.set_index(quantity_x)
    arr_y = []
    if (quantity_y in ["profit_std", "semideviation", "VaR", "ES", "probab_loss"]):
        for i in range(len(arr_x)):
            max_y = df0.loc[arr_x[i], quantity_y].min()
            arr_y.append(max_y)
    else:
        for i in range(len(arr_x)):
            max_y = df0.loc[arr_x[i], quantity_y].max()
            if (abs(max_y) < 0.0000000000001): max_y = np.float(0)
            arr_y.append(max_y)
    arr_y = np.array(arr_y)

    del datafile_path;del quantity_x;del quantity_y; del df0

    return arr_x, arr_y, True

# -----------------------------------------------------------------------------------------------

def plot_optima_2D(datafile_name, output_dir, plot_name, quantity_x, quantity_y, quantity_y2=None, rescaling_y2=1.000):
    '''This function makes a 2D plot whose variable x is "quantity_x" and whos variable y is the optimum (maximum or
     minimum of "quantity_y").'''

    from module_parameters import distribution_types_text

    # Reading the data
    arr_x, arr_y, succeeded = read_data(datafile_name, quantity_x, quantity_y)
    if (not succeeded): return
    text = ""

    # Actual plotting
    fig, ax = plt.subplots()
    plt.xlabel(plot_labels[quantity_x], fontsize=16)
    if (plot_labels[quantity_y]=="$SR$"): my_y_label = "Sharpe ratio"
    else: my_y_label = plot_labels[quantity_y]
    plt.ylabel(my_y_label, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.tick_params(axis='both', which='minor', labelsize=11)
    plt.xticks(rotation=45, ha='right')

    aux_string = datafile_name.split(sep="/")[-1]
    aux_string=aux_string.replace("OrnUhl-spr_resid_",""); aux_string=aux_string.replace(".csv","")
    for distr in ["-norm","-nct","-levy_stable","-genhyperbolic"]:
        if distr in aux_string:
            aux_string1 = aux_string.replace(distr,"")
            aux_string2 = distr.replace("-","")
            break

    if ("x_vs_y" in aux_string1):
        aux_string1 = aux_string1.replace("_x_vs_y","")
        label_x = aux_string1.split(sep="_")[1]
        label_y = aux_string1.split(sep="_")[0]
        my_title = label_x + " / " + label_y + "; "
    elif ("y_vs_x" in aux_string1):
        aux_string1 = aux_string1.replace("_y_vs_x","")
        label_x = aux_string1.split(sep="_")[0]
        label_y = aux_string1.split(sep="_")[1]
        my_title = label_x + " / " + label_y + "; "
    else:
        my_title = ""
    my_title += distribution_types_text[aux_string2] + " distribution"
    ax.set_title(my_title, fontsize=16)

    color1 = 'b'
    color2 = '#8b0000'

    # color1 ='#006400'; color2 = '#ff4500'
    # plt.text(1.10, 0.0144, r'$\it{With \,\, Poisson \,\, events}$', fontdict={'size': 16})
    # plt.ylim([0.007,0.015])
    # plt.text(1.005, 0.114, r'$\it{Without \,\, Poisson \,\, events}$', fontdict={'size': 16})
    # plt.ylim([0.04,0.12])

    ax.plot(arr_x, arr_y, '.', label=plot_labels[quantity_y], color=color1, linewidth=2, markersize=10)
    #plt.xlim([arr_x.min() - abs(arr_x.min() - arr_x.max()) / 100, arr_x.max() + abs(arr_x.min() - arr_x.max()) / 100])
    lim_0 = arr_x.min() - abs(arr_x.min() - arr_x.max()) / 100
    lim_1 = arr_x.max() + abs(arr_x.min() - arr_x.max()) / 100
    if (lim_0 < -0.00000001):
        lim_1 = max(lim_1,0)
    if (lim_1 > 0.00000001):
        lim_0 = min(lim_0,0)
    plt.xlim([lim_0,lim_1])

    try:
        x_smooth = np.linspace(arr_x.min(), arr_x.max(), 300)
        y_smooth = make_interp_spline(arr_x, arr_y)(x_smooth)
        ax.plot(x_smooth, y_smooth, '-', color=color1, linewidth=2)  # 'b', '#006400'
        del y_smooth;
        del x_smooth
    except ValueError:
        print("\nWARNING: Unable to perform the interpolation of " + quantity_y + "-vs-" + quantity_x + "\n")

    if (quantity_y2 != None):
        arr_x2, arr_y2, succeeded = read_data(datafile_name, quantity_x, quantity_y2)
        arr_y2 *= float(rescaling_y2)
        my_label2 = plot_labels[quantity_y2]
        if (quantity_y2 == "Sharpe_ratio_with_semideviation"): my_label2 = ("$SR'$")#("Sharpe ratio from\nsemi-deviation " + r'($\times 1/\sqrt{2}$' + ")")
        ax.plot(arr_x2, arr_y2, "s", label=my_label2, color=color2, linewidth=2, markersize=5)
        text = "-and-" + str(quantity_y2)
        try:
            x2_smooth = np.linspace(arr_x2.min(), arr_x2.max(), 300)
            y2_smooth = np.array(make_interp_spline(arr_x2, arr_y2)(x2_smooth))
            ax.plot(x2_smooth, y2_smooth, '--', color=color2, linewidth=2)
            del x2_smooth;
            del y2_smooth
        except ValueError:
            print("\nWARNING: Unable to perform the interpolation of " + quantity_y2 + "-vs-" + quantity_x + "\n")

        del arr_x2;
        del arr_y2;

    if (( label_y == "BTC-USD") or ( label_x == "BTC-USD")):
        my_label_loc = 'center right'
    else:
        if ('stable' in datafile_name): my_label_loc = 'best'
        else: my_label_loc = 'lower center'
    plt.legend(loc=my_label_loc, bbox_to_anchor=(0.5, 0.1, 0.5, 0.5), labelspacing=1.5)
    plot_path = output_dir + "/Plots/" + plot_name
    plot_path = plot_path.replace("OrnUhl-E0_", "Opt_" + str(quantity_y) + text + "-vs-" + str(quantity_x) + "_OrnUhl-E0_")
    plot_path = plot_path.replace("OrnUhl-spr_resid_","Opt_" + str(quantity_y) + text + "-vs-" + str(quantity_x) + "_OrnUhl-E0_")
    plt.savefig(plot_path + ".pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    del quantity_x;  del quantity_y; del ax; del fig; del arr_y; del arr_x; del my_title; del aux_string2; del aux_string1;

    return


# -----------------------------------------------------------------------------------------------

def plot_from_several_files( list_input_file_paths, list_fields, output_dir, plot_name ):
    '''This function makes a 2D plot for several input files'''

    my_colors = ['b', '#8b0000', '#006400', '#00ff00', '#0000cd', '#666666', '#801414', '#00008b', '#ee82ee', '#8060c0',
                 '#804014', '#00ffff', '#c04000', '#ff7f50', '#ffd700', '#00c000', 'r', 'c', 'm', 'y', 'k', ]
    my_linestyles = ['solid', 'dotted', 'dashed', 'dashdot', '(0,(1,10))', '(0,(1,1))', '(5,(10,3))', '(0,(5,10))',
                     '(0,(5,5))', '(0,(3,10,1,10))', '(0,(3,5,1,5))', '(0,(3,1,1,1))', '(0,(5,5,1,5,1,5))',
                     '(0,(3,1,1,1,1,1))']

    fig, ax = plt.subplots()
    ax.legend(loc='center right')

    i = 0
    for filepath in list_input_file_paths:
        for fields in list_fields:

            quantity_x = fields[0]
            quantity_y = fields[1]
            arr_x, arr_y, succeeded = read_data(filepath, quantity_x, quantity_y)
            if (not succeeded): continue

            ax.plot(arr_x, arr_y, '.', label=plot_labels[quantity_y], color=my_colors[i % len(my_colors)], linewidth=2,
                    markersize=11)
            i += 1

            try:
                x_smooth = np.linspace(arr_x.min(), arr_x.max(), 300)
                y_smooth = make_interp_spline(arr_x, arr_y)(x_smooth)
                ax.plot(x_smooth, y_smooth, my_linestyles[i % (len(my_linestyles))],
                        color=my_colors[i % len(my_colors)], linewidth=2)
                i += 1
                del y_smooth;
                del x_smooth
            except ValueError:
                print("\nWARNING: Unable to perform the interpolation of " + quantity_y + "-vs-" + quantity_x + "\n")

    plot_path = output_dir + "/Plots/" + plot_name
    plt.savefig(plot_path + ".pdf", format="pdf", bbox_inches="tight")
    plt.clf()

    del ax;
    del fig;
    del list_input_file_paths;
    del list_fields;
    del output_dir;
    del plot_name;
    del arr_x;
    del arr_y;
    del x_smooth;
    del y_smooth;
    del i

    return

# -----------------------------------------------------------------------------------------------

def make_plots_convergence( input_params, product_label, distrib_type ):
    '''This function plots the convergence files stored in the Output/Output_trading_rules folder.'''

    from os import listdir, path
    from matplotlib import cm

    print("\n * Now making convergence plots (find them at "+input_params.dir_trading_rules_convergence + "/Plots).\n")

    # BLOCK 1: GENERATING THE DATA

    if not path.exists(input_params.dir_trading_rules_convergence + "/Plots"): makedirs(input_params.dir_trading_rules_convergence + "/Plots")
    #list_files0 = listdir( input_params.dir_trading_rules_convergence )
    list_files0 = pd.read_csv(input_params.dir_trading_rules_convergence + "/list_files_to_analyse_for_convergence.csv")
    list_files0 = list_files0["filepath"].values.tolist()
    if (len(list_files0)==0):
        print("\n WARNING: The file "+input_params.dir_trading_rules_convergence + "/list_files_to_analyse_for_convergence.csv is empty; no convergence plots can be made.\n")
    list_files = []
    for filename in list_files0:
        prefix = product_label.replace("spr_resid_",""); prefix = prefix.replace(".csv","")
        if ( (prefix in filename) and (not ("_convergence_sweep" in filename))):
            list_files.append(filename)
    print("Now reading", list_files[0])
    print(pd.read_csv( list_files[0]))
    df0 = pd.read_csv( list_files[0], header=0)
    max_len = len(df0)
    if (max_len<3000):
        print("\n WARNING: Insufficient data ("+str(max_len)+") for convergence analysis.\n")
        del input_params; del product_label; del df0; del list_files; del list_files0; del max_len
        return
    list_sizes_to_plot = []
    list_paths_to_plot = []
    list_aux = [1000,3000,10000,20000,30000,100000,300000,1000000,3000000,10000000,30000000,100000000,300000000,1000000000,3000000000,10000000000]
    for size in list_aux:
        if size <= max_len:
            list_sizes_to_plot.append(size)

    for filename in list_files:

        df0 = pd.read_csv(  input_params.dir_trading_rules_convergence + "/" + filename, header=0 )
        dfheader = (list(df0)[0])
        str1 = dfheader.replace("profit_#","")
        str1 = str1.split(sep="__")
        ev = np.float( str1[0].replace("ev_","") )
        pt = np.float(str1[1].replace("pt_", ""))
        sl = np.float(str1[2].replace("sl_", ""))
        plot_label = "ev="+"{:.3f}".format(ev) + "; pt="+"{:.3f}".format(pt)
        if (abs(sl) < 100): plot_label += "; sl="+"{:.3f}".format(sl)

        df_res = pd.DataFrame(index=list_sizes_to_plot,columns=["avg_profit","semideviation_profit","stdev_profit","Sharpe_ratio","Sharpe_ratio_from_semideviation","plot_label"])
        df_res["plot_label"]=plot_label
        df_res.index.names = ["N_MC_paths"]
        for size in list_sizes_to_plot:
            my_array = df0.loc[0:size,dfheader].to_numpy()
            my_array_mean = np.mean(my_array); my_array_stdev = np.std(my_array)

            # Calculation of semi-deviation
            semideviation = 0.0
            len_array = len(my_array)
            for i in range(len_array):
                if (my_array[i] < my_array_mean):
                    semideviation += ( my_array[i] - my_array_mean) ** 2
            semideviation = np.sqrt(semideviation / len_array)

            df_res.loc[size,"avg_profit"] = my_array_mean
            df_res.loc[size, "stdev_profit"] = my_array_stdev
            df_res.loc[size, "semideviation_profit"] = semideviation
            df_res.loc[size, "Sharpe_ratio"] = my_array_mean/my_array_stdev
            df_res.loc[size, "Sharpe_ratio_from_semideviation"] = my_array_mean/semideviation

            #print(size,",",my_array_mean,",",my_array_stdev,",",my_array_mean/my_array_stdev,",",my_array_mean/(np.sqrt(2)*semideviation) )
        
        filepathout = (input_params.dir_trading_rules_convergence + "/" + filename).replace(".csv","_convergence_sweep.csv")
        list_paths_to_plot.append(filepathout)
        df_res.to_csv( filepathout,index=True)
        del df_res

    list_files0 = listdir( input_params.dir_trading_rules_convergence ); list_paths_to_plot=[]
    for filepath in list_files0:
        if ("sweep.csv" in filepath): list_paths_to_plot.append(filepath)


    # BLOCK 2: MAKING THE PLOTS

    mycolor = cm.rainbow(np.linspace(0, 1, len(list_paths_to_plot)))
    axes_text = {"avg_profit":"average profit","stdev_profit":"standard dev. profit","semideviation_profit":"Semideviation of profit","Sharpe_ratio":"$SR$","Sharpe_ratio_from_semideviation":"$SR'$"}
    for quantity_to_plot in ["avg_profit","stdev_profit","semideviation_profit","Sharpe_ratio","Sharpe_ratio_from_semideviation"]:
        fig, ax = plt.subplots()
        max_y = -999999; min_y = 9999999
        count=0
        for datapath in list_paths_to_plot:
            df0 = pd.read_csv(input_params.dir_trading_rules_convergence +"/" +datapath,header=0)
            arr_x = np.log10(df0["N_MC_paths"])
            if (quantity_to_plot=="Sharpe_ratio_from_semideviation"):
                arr_y = (df0[quantity_to_plot])/np.sqrt(2)
            else:
                arr_y = df0[quantity_to_plot]
            max_y = max( max_y, max(arr_y[5:]) )
            min_y = min( min_y, min(arr_y[5:]) )
            ax.tick_params(axis='both', which='major', labelsize=11)
            ax.tick_params(axis='both', which='minor', labelsize=11)
            plt.xticks(rotation=45, ha='right')
            ax.plot(arr_x, arr_y, '-', color=mycolor[count], linewidth=1.2, label = df0["plot_label"].at[0] )
            ax.plot(arr_x, arr_y, '.', color=mycolor[count], markersize=4)
            count += 1

        plt.xlabel("$N_{paths}$", fontsize=16)
        plt.ylabel(axes_text[quantity_to_plot], fontsize=16)
        plt.xticks((np.log10(1000),np.log10(3000),np.log10(10000),np.log10(20000),np.log10(30000),np.log10(100000),np.log10(300000),np.log10(1000000),np.log10(3000000)), (1000,3000,10000,20000,30000,100000,300000,1000000,3000000))
        plt.ylim(min_y * 0.992, max_y * 1.005)
        mytitle = rewrite_title( product_label )
        mytitle = "Convergence (" + mytitle.replace("Residuals of spreads of","")+" )"
        ax.set_title(mytitle, fontsize=16)

        # if (quantity_to_plot=="Sharpe_ratio_from_semideviation"): plt.ylim( 1.230, 1.255)
        # if (quantity_to_plot=="avg_profit"): plt.ylim(0.0696,0.0721)
        # if (quantity_to_plot == "semideviation_profit"): plt.ylim(0.03975, 0.04165)

        plt.xlim(np.log10(10000), np.log10(np.max(list_sizes_to_plot)))
        plt.legend(loc='best',labelspacing=0.4)  # loc='lower right', loc='best', bbox_to_anchor=(0.5, 0.1, 0.5, 0.5),
        filepath_plot = input_params.dir_trading_rules_convergence + "/Plots/convergence_" + product_label + "_" + distrib_type + "_" +quantity_to_plot+ ".pdf"
        plt.savefig(filepath_plot, format="pdf", bbox_inches="tight")
        plt.clf(); plt.cla(); plt.close('all')

    del input_params; del product_label; del list_paths_to_plot; del list_files; del list_files0; del list_aux

    return

# -----------------------------------------------------------------------------------------------

