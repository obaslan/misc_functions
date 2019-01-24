#!/usr/bin/env python
# coding: utf-8
# Misc. Functions
running:  
F = open("Library_address.txt",'r') 
Library_address = F.read()  
%run $Library_address/misc_functions.ipynb
# In[1]:


import numpy as np
import scipy.integrate as integrate
import scipy
from scipy import signal
from scipy import interpolate
from pandas import rolling_median, rolling_mean
import math
exp=math.exp
pi=math.pi
inf=math.inf

# Default folder,file properties
DF = open("Directory_address.txt",'r') 
Directory_address = DF.read()

folder_address = r"C:\Users\Aslan\Desktop\Pyhton_Coding\Inputs"
folder_address = Directory_address + "\Inputs"

output_folder_address = r"C:\Users\Aslan\Desktop\Pyhton_Coding\Outputs"
output_folder_address = Directory_address + "\Outputs"
def_file_name= "data"
file_labels = ""
data_type= ".txt"


# In[7]:


# Animated plots
#http://tiao.io/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/

from IPython.display import HTML
from matplotlib.animation import FuncAnimation
def plot_animation(data):
    fig, ax = plt.subplots(figsize=(5, 3))
    #ax.set( xlim=(1.4, 1.9), ylim=(-40, data.max() ) )
    ax.set(ylim=(-40, data.max() ) )

    line = ax.plot(data[:,0], data[:,1], color='k', lw=2)[0]

    anim = FuncAnimation(
        fig, animate, interval=1000, frames = range(1, len(data[0,:])), fargs= (data.T,line) )

    return HTML(anim.to_html5_video()) , HTML(anim.to_jshtml())

def animate(i, *fargs):
    y = fargs[0]
    line = fargs[1]
    line.set_ydata(y[i])
    


# In[1]:


def despike(data, window = 20, n_deviation = 0.35):


    df = pd.DataFrame(data)
    data_median_pd = df.copy()
    data_diff_pd = df.copy()
    outlier_idx =  df.copy()
    data_despiked = df.copy()
    row,column = data.shape
    
    for n in range(1,column):
        data_median_pd[n] = rolling_median(df[n], window=window, center=True).fillna(method='bfill').fillna(method='ffill')
        data_diff_pd[n] = np.abs(df[n] - data_median_pd[n]) # deviation from the rolling median

        outlier_idx[n] = data_diff_pd[n] > n_deviation*df[n].std() # spike: if the deviation is much more than the global standard deviation

        data_despiked[n][outlier_idx[n]] = data_median_pd[n][outlier_idx[n]] # replaced the spikes with the rolling median values

        if outlier_idx[n].any(): # plot if any spike is detected
            # the following plots most of the relavent calculations
            #smart_plot([  df[0], df[0][outlier_idx[n] ], df[0], df[0], df[0],df[0] ], [ df[n], df[n][outlier_idx[n]], data_median_pd[n], data_diff_pd[n], df_mad, data_despiked[n] ], x_label='Energy (eV)', y_label='PL',
            #          label = ['data', 'spikes', 'median', 'data-median', 'deviation', 'despiked' ],lines=['-','x','--', '-','-','-'], ms=[1,15,1,1,1,1], annotate = 3, figsize=(16,10))


            smart_plot([ df[0], df[0]  ], [ df[n],  data_despiked[n] ], legend_title='Removing Spikes',label = [data_labels[n], 'After removal'], x_label='Energy (eV)', y_label='PL', lines=['-','-'], ms=[1,15], annotate = 0)

    plt.show()
    return data_despiked


# In[6]:


def correct_eff(data, eff, data_labels=[], eff_labels=[], file_name='', material='',
                x_normalize = 1.60, # x value at which the inverse efficiency is 1
                x_lower = 1.4, x_upper = 1.9
    ):
    
    
    x_normalize = {'MoSe2': 1.50 ,'MoS2': 1.80, 'WSe2':1.60, 'WS2': 1.95 }.get(material, x_normalize)
    
    # auto-assign the limits if the material is specified
    x_lower, x_upper = {'MoSe2': [1.35, 1.8] , 'WSe2': [1.3, 1.9],'WS2': [1.8, 2.2] }.get(material, [x_lower, x_upper])
    
    ## interpolate the efficiency range to the range of the data imported
    # create a new np_array of 2 columns for interpolated efficiency
    eff_interp=np.zeros(shape=(len(data[:,0]),2))

    # x values to interpolate for
    eff_interp[:,0] = data[:,0]

    eff_interp[:,1]=interpolate.interp1d(eff[:,0], eff[:,1])(eff_interp[:,0])
    smart_plot([eff_interp[:,0],eff[:,0]],[eff_interp[:,1],eff[:,1]], y_label='Collection Efficiency', label=['Interpolated ','Whole Range'], lw=[4,2],figsize=(10,3), legend_title=eff_labels[1])   
    
    
    # assign the inverse collection efficiency to a separate list    
    idx = (np.abs(eff_interp[:,0] - x_normalize)).argmin() # the index where x_normalize is

    normal_inverse_eff=eff_interp[idx,1]/eff_interp[:,1] 
    smart_plot([eff_interp[:,0]],[normal_inverse_eff], x_label='Energy (eV)', y_label='Normalized Inverse \n Collection Efficiency', figsize=(9,4))
    
    # Correct the data for the collection efficiency
    row,column = data.shape

    data_eff_corrected = np.zeros_like(data)

    m=1 # Column to plot before and after interpolation
    data_eff_corrected[:,0] = data[:,0]
    for n in range(1,column):
        data_eff_corrected[:,n]=np.multiply(normal_inverse_eff, data[:,n])
    #custom_limit_x=1, x_min=1.4, x_max=1.9,
    smart_plot([data[:,0],data_eff_corrected[:,0]], [data[:,m],data_eff_corrected[:,m]],label=[data_labels[m],"Efficiency Corrected"], custom_limit_x=1, x_min=x_lower, x_max=x_upper,  figsize=(9,4))
    plt.show()
    
    
    if file_name != '':
        export_np(data_eff_corrected, file_name= 'data_eff_corrected_' + file_name ) # do not export the smoothed data
    return data_eff_corrected


# In[8]:


def subtract_ref (data, data_labels=[], ref=[['']], ref_labels=[], file_name='',
                  x_lower = 1.4, x_upper = 1.9, material=''):
    
    # auto-assign the limits if the material is specified
    x_lower, x_upper = {'MoSe2': [1.35, 1.8] , 'WSe2': [1.3, 1.9],'WS2': [1.8, 2.2] }.get(material, [x_lower, x_upper])

    row,column = data.shape
    data_ref_subtracted = np.zeros_like(data)
    data_ref_subtracted[:,0]= data[:,0]
    
    if ref[0][0]: # if a reference is provided
        # interpolate the reference range to the range of the data imported
        # create a new np_array of 2 columns for interpolated reference
        ref_interp = np.zeros(shape=(len(data[:,0]),2))

        # x values to interpolate for
        ref_interp[:,0] = data[:,0]
        ref_interp[:,1] = interpolate.interp1d(ref[:,0], ref[:,1])(ref_interp[:,0])

        smart_plot([ref_interp[:,0],ref[:,0]],[ref_interp[:,1],ref[:,1]], label=['interpolated '+ref_labels[1],'Reference'], lw=[4,2],figsize=(12,4))

        # Correct the data by subtracting the reference signal
        for n in range(1,column):
            data_ref_subtracted[:,n]=np.subtract(data[:,n], ref_interp[:,1])
            smart_plot([data[:,0],data_ref_subtracted[:,0]], [data[:,n],data_ref_subtracted[:,n]],label=[data_labels[n],"Reference Subtracted"],  custom_limit_x=1, x_min=x_lower, x_max=x_upper,autoscale_view_y=0)

    else : # no reference is provided
        # Correct the data by subtracting a linear baseline signal (obtained by fitting 2 points (averaged over 10 points each) of data to a line)

        m=5 # Column to plot before and after interpolation
        if m > column:# not as many columns in the data
            m = 1

        x_lower_index = find_nearest(data[:,0], x_lower)[0]
        x_upper_index = find_nearest(data[:,0], x_upper)[0]

        for n in range(1,column):
            baseline = scipy.polyfit( [x_lower, x_upper], [ np.average(data[x_lower_index-5:x_lower_index+5,n]) , np.average(data[x_upper_index-5:x_upper_index+5, n]) ], 1 )
            data_ref_subtracted[:,n] = data[:,n] - data[:,0]*baseline[0]-baseline[1]
            smart_plot([data[:,0]]*3 + [[x_lower-0.1,(x_lower+x_upper)/2, x_upper+0.1]], [ data[:,n], data_ref_subtracted[:,n], data[:,n]-data_ref_subtracted[:,n], [0,0,0] ], label=[data_labels[n],"After Subtraction", 'Baseline (%.2fx+%.2f)' %(baseline[0], baseline[1]), 'Zero'],  custom_limit_x=1, x_min=x_lower, x_max=x_upper)
    plt.show()
    if file_name != '':
        export_np(data_ref_subtracted, file_name='data_ref_subtracted_'+ file_name)
        
    return data_ref_subtracted


# In[8]:


def smoother(data, data_labels, smooth_window_length = 9, polyorder = 2,
             x_lower = 1.4, x_upper = 1.9, material=''):
    
    row,column = data.shape
    data_smooth = np.zeros_like(data)
    data_smooth[:,0] = data[:,0]
    
    # auto-assign the limits if the material is specified
    x_lower, x_upper = {'MoSe2': [1.35, 1.8] , 'WSe2': [1.3, 1.9],'WS2': [1.8, 2.2] }.get(material, [x_lower, x_upper])


    
    
    n=1
    for n in range(1,column):
        data_smooth[:,n]=scipy.signal.savgol_filter([data[:,0],data[:,n]], smooth_window_length, polyorder, deriv=0, axis=1)[1]
        #smart_plot( [data_smooth[:,0], data_smooth[:,0], [x_min, x_max]], [data[:,n], data_smooth[:,n] , [0,1] ], label=[data_labels[n], 'data_smooth', '0 line'], lw=[4,2,.5], figsize=(9,3),  custom_limit_x=1, x_min=x_min, x_max=x_max)
        #, custom_limit_x=1, x_min=x_lower, x_max=x_upper
        smart_plot( [data_smooth[:,0], data_smooth[:,0], [x_lower, (x_lower+x_upper)/2, x_upper]], [data[:,n], data_smooth[:,n] , [0,0,0] ], label=[data_labels[n], 'data_smooth', '0 line'], lw=[4,2,.5], figsize=(9,3), custom_limit_x=1, x_min=x_lower, x_max=x_upper)

    plt.show()
    return data_smooth


# In[3]:


#TODO : Plot vertical lines at peak positions
def peak_analyzer(data, x_lower=0, x_upper=4, is_print_report = 0, data_labels=[''], file_name='', pressures=[''], diameters=[''],
                 material='', 
                is_map=[0,0,0] # boolean for x,y,z coordinates if mapping
                 ):
    #if len(x) is not len(y) :
       # print('x and y do not have the same length')
        #return
    # auto-assign the limits if the material is specified
    x_lower, x_upper = {'MoSe2': [1.35, 1.8] , 'WSe2': [1.3, 1.9],'WS2': [1.8, 2.2] }.get(material, [x_lower, x_upper])
    # typical, good linewidth
    typ_linewidth = {'MoSe2': 40 ,'MoS2': 50, 'WSe2': 42, 'WS2': 30 }.get(material, 30)
    
    header = ['Peak Energy', 'X half-left', 'X half-right', 'Linewidth', 'Asymmetry', 'Max Amplitude', 'Integ. Area']
    result = []    
    for n in range(1, len(data[1,:])): # for all y columns of the data
        
        x = data[:,0] # reassign the x each time to have it in numpy format which works in the select_range() method
        y = data[:,n]
    
        x, y = select_range(x, y, x_lower, x_upper)

        x = list(x)
        y = list(y)

        Area=np.trapz(y,x)
        ymax=max(y)
        ymin=min(y)

        N=len(y)
        lev50 = ymax/2.0

        if y[0] < lev50 :                # find index of center (max or min) of pulse
            centerindex= y.index(ymax)
            #centerindex, value = max(y, key=lambda y: y[1])
            xmax = x[centerindex]
            Pol = +1
            if is_print_report:
                print('Pulse Polarity = Positive')
        else :
            centerindex= y.index(ymin)
            xmin = x[centerindex]
            Pol = -1
            if is_print_report:
                print('Pulse Polarity = Negative')

        i = 2
        while (np.sign(y[i]-lev50) == np.sign(y[i-1]-lev50)) :
            i = i+1
            #first crossing is between y(i-1) & y(i)
        interp = (lev50-y[i-1]) / (y[i]-y[i-1])
        tlead = x[i-1] + interp*(x[i]-x[i-1])

        i = centerindex+1                  #start search for next crossing beyond the center
        while ((np.sign(y[i]-lev50) == np.sign(y[i-1]-lev50)) and (i <= N-1)):
            i = i+1

        if i is not N :
            if is_print_report:
                print('Pulse is Impulse or Rectangular with 2 edges')
            interp = (lev50-y[i-1]) / (y[i]-y[i-1])
            ttrail = x[i-1] + interp*(x[i]-x[i-1])
            width =  ttrail-tlead
        else:
            Ptype = 2
            if is_print_report:
                print('Step-Like Pulse, no second edge')
            ttrail = 'NaN'
            width = 'NaN'

        asymmetry =(xmax-ttrail)/(tlead-xmax)
        result.append([xmax, ttrail, tlead, abs(width*1000), asymmetry, ymax, abs(Area)]) #label=[data_labels[n], 'Width %.1fmeV' %width*1000, 'Peak %.3f eV' %xmax]
        smart_plot( [x, [ttrail, tlead], [ttrail, tlead] ], [ y, [ymax/2,ymax/2], [ymax,ymax] ], label=[data_labels[n], '%.1f meV (Width)' %abs(width*1000), '%.3f eV (Peak)' %xmax] , lw=[4,2,.5], figsize=(9,3) , v_lines=[xmax], autoscale_view_y=0 )#, custom_limit_x=1, x_min=x_lower, x_max=x_upper
                  
        plt.show()
    units = ['eV', 'eV', 'eV', 'meV', '', '', '']
    index = pd.MultiIndex.from_arrays([header, units])
    fit_report = pd.DataFrame(result, index=data_labels[1:], columns=index)


    if is_map[0]: #x coordinate
        fit_report.insert(0, 'x', [item[0] for item in coordinates])
    if is_map[1]: #y coordinate
        fit_report.insert(1, 'y', [item[1] for item in coordinates])
    if is_map[2]: #z coordinate
        fit_report.insert(2, 'z', [item[2] for item in coordinates])
        
    #fit_report.columns.set_levels( ['1','2'] + units, level=1,inplace=True)
        
    # Add a column for futher calculations
    #if not all(pressure == '' for pressure in pressures):
    #    psi = 6894.744825 # 1 psi in N/m^2
    #    micron = 10**-6 # 1 micrometer in meters
    #    PR_2_3rd = [(pressure*psi* diameter*micron/2)**(2/3) for (pressure,diameter) in zip(pressures,diameters)]
    #    fit_report.insert(0, '(Pressure x Radius)\+(2/3)', PR_2_3rd)
    
    if file_name != '':
        #export_np(header, file_name='analysis_'+ file_name, header= '\t'.join(header))
        export_pd(fit_report, file_name='analysis_'+ file_name, index=True, header=True)
        

    for j in range(len(result[0])):
        fit_report[header[j]] = [item[j] for item in result]
    
    #fit_report.columns=fit_report.columns.str.replace('#','') 
    
    
    fig = plt.figure()
    plt.plot(fit_report['Peak Energy','eV'], fit_report['Linewidth'], label='A exciton', marker='o', markersize=12, color="red")
    plt.plot( [min(fit_report['Peak Energy','eV']), max(fit_report['Peak Energy','eV'])  ], [typ_linewidth]*2, ls='--', label= material+' typical', color="blue")
    plt.xlabel('Peak Energy (eV)')
    plt.ylabel('Linewidth (meV)')
    plt.legend()
    fig.set_size_inches(w=8,h=6)
    plt.show()
    
    pd.options.display.float_format = '{:,.3f}'.format # display 3 decimals          
    return fit_report #,['%.3f' % i for i in result]


# In[ ]:


# Remove the zero columns of data and its labels, export the nonzero data and return it and the nonzero labels
def remove_zero_columns(data, data_labels=[], file_name=''):
    row,column=data.shape
    zero_columns = []

    for n in range(column):
        if not any(data[0:2,n]) : # if 1st 3 rows are 0
            zero_columns.append(n)   
    
    n_zero_columns = len(zero_columns) # total number of zero columns
            
    print("%d nonzero + %d zero = %d columns" %(column-n_zero_columns, n_zero_columns, column))
    
    data_nonzero = np.delete(data, zero_columns, 1)
    if file_name != '':
        export_np(data_nonzero, file_name='data_nonzero_'+file_name)
        
    if data_labels: 
        data_nonzero_labels = list(np.delete(data_labels, zero_columns)) 
        return data_nonzero, data_nonzero_labels,n_zero_columns
    else:
        return data_nonzero,n_zero_columns    


# In[19]:


def evenly_space(x, y, step_size=0.001, # step size of the new x values to be interpolated
                 is_plot = 0 #
               ): 
    
    interp=[] # interpolator
    column = len(y[0])
    
    label=['interpolated','data']; lw=[4,2]
    kwargs = {'title' : '', 'legend_title' : legend_title, 'label': label, 'legend_fs' : legend_fs, 'lines' : lines, 'lw' : lw, 'ms' : ms, 'x_label' : x_label, 'y_label' : '1', 'figsize' : (12,4)
         ,'is_x_numeric' : 1}
    
    # x values to interpolate for
    x_interp = np.arange(min(x), max(x), step_size)

    y_interp=np.zeros(shape=(len(x_interp),column))

    for n in range(0, len(y[0,:])): # for all y columns of the data
        interp.append(interpolate.interp1d(x, y[:,n]))
        
        kwargs['legend_title'] = data_labels[n] ; 
        y_interp[:,n]=interp[n](x_interp)  
        if is_plot:
            smart_plot([x_interp, x],[y_interp[:,n], y[:,n]],  **kwargs)
    if is_plot:
        plt.show()
    return x_interp, y_interp


# In[20]:


def smooth_deriv(x, y, deriv=1, # derivative order
                 polyorder=1, # polynomial order for smoothing
                 derive_length=3, # smoothing window length for derivative
                 smooth_length=3 # smoothing window length after derivative
                ):
    
    x, y = evenly_space(x, y) # to properly derive
    
    y_derive = np.zeros_like(y)
    y_derive_smooth = np.zeros_like(y)
    
    x_min=min(x); x_max=max(x); lw=[4, 2, 0.5]; lines = ['k-','y-', 'r-']; label = ['Derivative', '2nd Smooth', '0 line']
    
    if deriv ==1:
        y_label = r'${\rm d/dE}$'
    else :
        y_label = r'$d^%s/dE^%s$' %(deriv,deriv)
    
    kwargs = {'x_min': x_min, 'x_max': x_max, 'y_label' : y_label  , 'title' : '', 'label': label, 'legend_title' : legend_title, 'legend_fs' : legend_fs, 'lines' : lines, 'lw' : lw, 'ms' : ms, 
              'figsize' : (12,4), 
         'is_x_numeric' : 1}
    
    for n in range(0, len(y[0,:])): # for all y columns of the data
        # Smooth and take derivative for all columns except the X-axis column (0th Column)
        y_derive[:,n] = scipy.signal.savgol_filter( [y[:,n]], derive_length, polyorder, deriv=deriv, axis=1)

        # smooth the 2nd derivative again since it worked better in ignoring false peaks
        y_derive_smooth[:,n]=scipy.signal.savgol_filter([y_derive[:,0],y_derive[:,n]], smooth_length, polyorder, deriv=0, axis=1)[1]
        
        kwargs['legend_title'] = data_labels[n]
        
        smart_plot( [x,x,[x_min,x_max]],[y_derive[:,n], y_derive_smooth[:,n], [0,0] ], **kwargs)

    plt.show()
    return  x, y_derive


# In[1]:


# Detect the zero crossings in a dataset
# may correspond to peaks, dips etc. in the actual data

def zero_cr(x, y, data_labels='', slope=0, zero=0, # see if crossing a non-zero value this way
            x_min=0, x_max=4, # ignore outside these x values if there are no peaks for sure
            is_plot=1, # whether to plot the 2nd derivatives and indicate the peak positions
           ):
    x_values=[] ; slope_values=[]; 
        
    x_temp=x[:-1]# drop the last element of the x axis since np.diff(y) is 1 element shorter than x and y
   
    for n in range(0, len(y[0,:])): # for all y columns of the data
       

        diff_sign_y = np.diff(np.sign( y[:,n]-zero), axis=0)
        diff_y = np.diff( y[:,n], axis=0)
        
        x_values_temp=[] ; slope_values_temp=[]
        
        for i in range(len(x_temp)):

            if (x_temp[i] > x_min) & ( x_temp[i] < x_max) & (diff_sign_y[i] > 0) & (diff_y[i] > slope):

                x_values_temp.append(x_temp[i])
                slope_values_temp.append(diff_y[i])
            
        x_values.append(x_values_temp)
        slope_values.append(slope_values_temp)
            
        # determine the A exciton x_value index (probably largest slope)
        index_max = np.argmax(slope_values[n])

        # determine the x_values[n] at a lower x_value than A exciton since we probably do not have any peaks there
        del_indices=[]
        for i in range(len(x_values[n])):
            if x_values[n][i] < x_values[n][index_max]:
                del_indices.append(i)
        # delete the elements at those indices starting from the highest index to avoid confusion
        for i in sorted(del_indices, reverse=True):
            del x_values[n][i]
            del slope_values[n][i]    
        #sorted x_slope_values[n] (j[0] is used to convert array[1] to 1)
        slope_values[n] = [ j for (k,j) in sorted((zip(x_values[n], slope_values[n])), key=lambda pair: pair[0])] # sort according to the peak positions
        x_values[n].sort() # sort in ascending order


        # plot (x,y) and mark the detected peak positions
        smart_plot([x, x_values[n], [min(x),max(x)]], [y[:,n], [i*10**-2 for i in slope_values[n]], [0,0]], label=[data_labels[n],'Peaks','Zero Line'], lw=[1,.4,0.3], figsize=(16,4)
                   ,lines=['-','x','-'], y_label='%s' %n, annotate = 1)
    plt.show()
        
    report = pd.DataFrame(x_values, columns=[i for i in range(1, len(max(x_values, key=len))+1)], index=data_labels)
    
    return report, x_values,slope_values #,slope_values[n] # sort the x_values[n] in ascending order, convert to list


# In[ ]:


# find the index of a numpy array with the closest value
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]


# In[2]:


# find the local max of a numpy array
def find_loc_max(x, y, x_lower, x_upper):
    x, y= select_range(x, y ,x_lower, x_upper)
    idx,max_y = find_nearest(y, max(y))
    return x[idx], max_y


# In[1]:


# round upto smallest integer multiple of base, such as 6,12,18 to 10,25,20 for base=5
def round_upto_base(x, base=5):
    return int(base * round(float(x)/base+0.5))


# In[7]:


# round down to smallest integer multiple of base, such as 6,12,18 to 5,10,15 for base=5
def round_to_base(x, base=5):
    return int(base * round(float(x)/base))


# In[ ]:


# round upto smallest (in log) power of base, such as 0.02 ,8, 89 to 0.01, 10, 100 if base=10 and return (20,-3), (8,1) (9,2)
def round_upto_base_power(x1,x2,base=5):
    x=x1-x2+10**-4 #add a small number in case x1=x2
    N=int(np.log(x)/np.log(base)+0.5) # 0.5 makes sure to round up
    return [int(x/base**(N-1)),N-1] # a format like [3,4] for x=350


# In[ ]:


def step(x):
    if x == 0:
        return 0.5
    return 0 if x < 0 else 1


# In[4]:


def step_lorentzian(x_upper,FWHM): # integration of a lorentzian broadened delta function
    return integrate.quad(lambda x:lorentzian(x, FWHM), -inf, x_upper)[0]


# In[3]:


def voigt(x, FWHM_L,FWHM_G, center=0, Amplitude=1, constant=0):
    return constant+Amplitude*integrate.quad(lambda xp:lorentzian(x-xp,FWHM_L,center)*gaussian(xp,FWHM_G,center), -inf, +inf)[0]


# In[2]:


def fwhm_voigt( FWHM_L,FWHM_G):
    return  0.5346*FWHM_L + (0.2166*FWHM_L**2+FWHM_G**2)**0.5


# In[1]:


def fwhm_gauss_voigt(FWHM_V,FWHM_L):
    return ((FWHM_V-0.5346*FWHM_L)**2-0.2166*FWHM_L**2)**0.5


# In[3]:


def fwhm_lorentz_voigt(FWHM_V,FWHM_G):
    a=0.5346
    b=0.2166
    coefficients=[a**2-b,-2*a*FWHM_V, FWHM_V**2-FWHM_G**2]
    FWHM_L= np.roots(coefficients)
    FWHM_L=[i for i in FWHM_L if i < FWHM_V]
    return FWHM_L[0]


# In[4]:


def thermal_voigt(x, FWHM_L,FWHM_G, kT=25, center=0, Amplitude=1, constant=0):
    infinite= 10*kT
    n=1
    return constant+Amplitude*exp(-(x/kT)**n)*integrate.quad(lambda xp:lorentzian(x-xp,FWHM_L,center)*gaussian(xp,FWHM_G,center), -infinite, +infinite)[0]


# In[4]:


def thermal_voigt2(x, FWHM_L,FWHM_G, kT=25, center=0, Amplitude=1, constant=0):
    infinite= 10*kT
    n=1
    return constant+Amplitude*integrate.quad(lambda xp:exp(-(xp/kT)**n)*lorentzian(x-xp,FWHM_L,center)*gaussian(xp,FWHM_G,center), -infinite, +infinite)[0]


# In[4]:


def thermal_lorentz(x, FWHM_L, kT=25, center=0, Amplitude=1, constant=0):
    infinite= 10*kT
    n=1
    return constant+Amplitude*integrate.quad(lambda xp:exp(-(xp/kT)**n)*lorentzian(x-xp,FWHM_L,center), -infinite, +infinite)[0]


# In[ ]:


# convert dielectric function to refractive index
def epsilon_to_n(ϵ):
    ϵ1 = ϵ.real; ϵ2 = ϵ.imag;
    ϵ = np.sqrt(ϵ1**2+ϵ2**2)
    n = np.sqrt((ϵ+ϵ1)/2) + 1j*np.sqrt((ϵ-ϵ1)/2) 
    return n


# In[3]:


# Refractive Index function out of the dielectric function parameters
def refractive(x, sigma, center=1, amplitude=1, constant=1, fwhm=0):
    fwhm = sigma*2
    ϵ = epsilon2(x, sigma=sigma, center=center, amplitude=amplitude, constant=constant)
    return epsilon_to_n(ϵ)


# In[4]:


# Imaginary part of this epsilon converges to the lorentzian lineshape when center>>FWHM
def epsilon(x, FWHM, center=1, Amplitude=1, constant=1):
    return constant+(center*Amplitude)*(2/pi)*(1/( x**2 - center**2 - 1j*x*FWHM))
def epsilon2(x, sigma, center=1, amplitude=1, constant=1, fwhm=0):
    FWHM = 2*sigma
    return constant+(center*amplitude)*(2/pi)*(1/( x**2 - center**2 - 1j*x*FWHM))


# In[4]:


def lorentzian(x, FWHM, center=0, Amplitude=1, constant=0):
    return constant+Amplitude*(2/pi)*(FWHM/( (2*x - 2*center)**2 + FWHM**2))


# In[30]:


def d_lorentzian__d_FWHM(x, FWHM, center=0, Amplitude=1):
    derivative = lambda FWHM: lorentzian(x, FWHM, center=0, Amplitude=Amplitude)
    return scipy.misc.derivative(derivative,FWHM)

def d_lorentzian__d_FWHM2(x, FWHM, center=0, Amplitude=1):
    L = lorentzian(x, FWHM, center=center, Amplitude=Amplitude)
    return L**2 * (pi/2) * ( (2*x/FWHM)**2 - 1) / Amplitude


# In[32]:





# In[2]:


def gaussian(x, FWHM, center=0, Amplitude=1, constant=0):
    sigma= FWHM/2.3548
    return constant+Amplitude/(sigma*(2*pi)**0.5 )*exp(-0.5*(x/sigma)**2)


# In[1]:


def select_range(x, y, x_lower_limit, x_upper_limit, is_monotonic = 1):
    
    
    if is_monotonic: # then just find the indices of x_lower_limit and x_upper_limit
        lower_index, temp = find_nearest(x, x_lower_limit)
        upper_index, temp = find_nearest(x, x_upper_limit)  
        
        if upper_index < lower_index:
            temp = upper_index
            upper_index = lower_index
            lower_index = temp
            
        x = x[lower_index:upper_index]
        y = y[lower_index:upper_index]
        
            
    else: # not monotonic, check all the values one by one

        x = x[(x_lower_limit < x) & (x < x_upper_limit)]
        y = y[(x_lower_limit < x) & (x < x_upper_limit)]

    return x, y


# In[9]:


def FWHM2(X,Y):
    half_max = max(Y) / 2.
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    #plot(X,d) #if you are interested
    #find the left and right most indexes
    left_idx = np.where(d > 0)[0]
    right_idx = np.where(d < 0)[-1]
    return (X[right_idx] - X[left_idx])[0] #return the difference (full width)


# In[16]:


# find the radius of curvature of a (x,y) plot at each x value
def radius_curvature(x,y):
    dx = np.gradient(x)
    dy = np.gradient(y)/dx
    ddy = np.gradient(dy)/dx
    return dy , ((1 + dy**2)**1.5)/ddy
    


# In[6]:


# MAY NOT BE IN USE (SEE MEMBRANE CALCULATIONS)
def radius_curvature_2(x,y):
    spl = scipy.interpolate.splrep(x,y,k=3) # no smoothing, 3rd order spline
    ddy = scipy.interpolate.splev(x,spl,der=2) # use those knots to get second derivative 
    return ddy


# In[10]:


# MAY NOT BE IN USE (SEE MEMBRANE CALCULATIONS)
#from scipy.interpolate import UnivariateSpline
import numpy as np

def curvature_splines(x, y=None, error=0.1):
    """Calculate the signed curvature of a 2D curve at each point
    using interpolating splines.
    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )
         or
         y=None and
         x is a numpy.array(dtype=complex) shape (n_points, )
         In the second case the curve is represented as a np.array
         of complex numbers.
    error : float
        The admisible error when interpolating the splines
    Returns
    -------
    curvature: numpy.array shape (n_points, )
    Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
    but more accurate, especially at the borders.
    """

    # handle list of complex case
    if y is None:
        x, y = x.real, x.imag

    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
    fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

    xˈ = fx.derivative(1)(t)
    xˈˈ = fx.derivative(2)(t)
    yˈ = fy.derivative(1)(t)
    yˈˈ = fy.derivative(2)(t)
    #curvature = (xˈ* yˈˈ - yˈ* xˈˈ) / np.power(xˈ** 2 + yˈ** 2, 3 / 2)
    curvature = (1 + yˈ**2)**1.5/yˈˈ 
    return curvature


# # Create file names for map measurements

# In[1]:


def map_labels(L_x, L_y, L_z=0, dL_x=10, dL_y=10, dL_z=10):
    labels=[]
    coordinates = []
    for j in range( int(L_y/dL_y) + 1):

        for i in range( int(L_x/dL_x) + 1):

            for k in range( int(L_z/dL_z) + 1):

                coordinates.append([i*dL_x, j*dL_y, k*dL_z]) 
                labels.append('x_%.1f_y_%.1f_z_%.1f' %(i*dL_x, j*dL_y, k*dL_z))
    return labels, coordinates


# In[5]:


map_labels(2,2, L_z=0, dL_x=1, dL_y=1, dL_z=10)


# # Change names of multiple files

# In[1]:


import os
# Add WSe2_ to the names of the files

with os.scandir(folder_address) as it:
    for entry in it:
        if entry.name.startswith("1L_A6_6_micron_#6a_0psi_PL") and entry.is_file():
            
            os.renames( os.path.join(folder_address, entry.name) ,  os.path.join(folder_address, 'WSe2_'+ entry.name))


# # Correct the Labeling for OriginPro

# In[13]:


def text_for_originpro(text_data, file=0, file_name='', folder_address='', output_folder_address='', print_OriginPro_commands=0):

    if file: # if a file is being corrected for originpro
        # Read the file
        input_address= folder_address + file_name + '.txt'

        with open(input_address, 'r') as file :
            text_data = file.read()
    
    # Replace the target strings
    # Put the commands in the priority order. I.e.; put 'A1s-A2s' before both 'A1s' and 'A2s'
      
    
    command_names = ['_micron', '_${\rm \mu}m$','_${\\rm \\mu}$m', '${\rm \mu}m$','${\\rm \\mu}$m'] # as imported from originpro OR used in python
    commands = [' \g(m)m']*5 # proper command in originpro
    originpro_names = [' μm']*5 # representation in originpro

    
    command_names.extend(['.asc'])
    commands.extend([''])
    originpro_names.extend([''])
      
    command_names.extend([' _PL', '_PL', ' _RC', '_RC'])
    commands.extend(['', '', '', ''])
    originpro_names.extend(['', '', '', ''])
    
    
    
    for i in range(len(commands)):
        # replace 'micron' with '\g(m)m'  (μm)
        if isinstance(text_data, list):
            text_data = [text.replace(command_names[i], commands[i]) for text in text_data]
        else :
            text_data = text_data.replace(command_names[i], commands[i]) 

    if file:# Write the file out again
        output_address = output_folder_address + file_name + '_corrected.txt'
        with open(output_address, 'w') as file:
            file.write(text_data)

    if print_OriginPro_commands:
        for (command,originpro_name) in zip(commands,originpro_names):
            if originpro_name:
                print(r" '%s' => '%s' in OriginPro" %(command, originpro_name))

    
    
    return text_data
        

