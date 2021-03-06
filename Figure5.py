'''
Created on 19. july 2017

@author: ELP
'''

import os,sys
import csv,time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import xarray as xr 
from scipy import interpolate
from util import path_brom1

plt.style.use('bmh') 
plt.rcParams['xtick.top'] =  True

'''    The module plot Figure 5 for the 
        Jellyfarm project, Hardangerfjord case , 2D BROM
        Data from the Brom are plotted from start_day to stop_day,
        For 2 columns: Baseline and "Farm"
'''
# Last modelled year 
#start_day = 731 + 167
#stop_day = 731 + 259 #1096
#baseline_col = 0
#farm_col = 19

# Change to True if you want to save figure 
save = True

# -- define fonts
small_size  = 8
medium_size = 10
bigger_size = 13
plt.rc('font',    size = medium_size)   # controls default text sizes
plt.rc('axes', labelsize = bigger_size)   # fontsize of the x and y labels

plt.rc('xtick', labelsize = bigger_size)   # fontsize of the tick labels
plt.rc('ytick', labelsize = bigger_size)   # fontsize of the tick labels
plt.rc('legend', fontsize = small_size)   # legend fontsize



#Observations
tic = [2190.56, 2021.71, 999.57, 988.85]
tic_depth = [470,200, 10,0]
alk = [2301.98,2180.317, 1058.218, 1059.57]

start_list = [0,7,12,17,23,28] 
end_list = [7,12,17,23,28,33]
st_list = [1,2,3,4,5,6]

#Observations data from sediments 
depth200 = [0.50, 1.20, 2.50 , 4.00, 7.5]
depth199 = [0.50, 1.50, 2.50 , 4.00, 7.5]

si200 = [285.7,357.1,328.6,300.0,314.3]
no3200 = [35.71428571, 16.42857143, 9.64, 8.92,2.857142857 ]
po4200 = [9.68, 9.68, 9.68, 8.39, 6.45]
tic200 = [2916.7, 3841.7, 3683.3, 3675.0, 4033.3]
alk200 = [2860, 3460, 3410, 3410, 3850]
nh4200 = [121.4285714,100.,100.,85.71428571,71.42857143]

si199 = [189.3, 171.4, 214.3, 175.0, 221.4]
no3199 = [3.928, 4.071, 5.571, 2.071, 2.714] 
po4199 = [41.94,30.97, 21.94,24.52, 10.32]
alk199 = [3070, 3050, 3590, 4040, 3800]
tic199 = [3141.7, 3641.7, 4191.7, 4133.3, 3916.7]
nh4199 = [100.,114.2857143,145.7142857,107.1428571,92.85714286]

colr1 = '#ff8800'
#colr_base = '#2e3236'
colr_base_field = '#519344'
colr_farm = '#c75641'
colr_mid = "#ff8f0f"
colr_mid2 = "#f4f441"
a = 0.4
ln  = 1.2
ln_field = 1

df = pd.read_csv('data/wat_hardan2.txt')

df_brom = xr.open_dataset(path_brom1)

depth_brom = df_brom['z'].values
sed_depth_brom  = (depth_brom - depth_brom[15])*100

# Add oxygen 
def get_df(path):
    d = pd.read_excel(path,skiprows = 1,usecols = [1,4,7],
                      names = ['depth','time','o2'])
    d.depth = d.depth / -10000
    #d = d.resample('30s', on='time').mean()  
    d.o2 = d['o2'].where(d.o2 >= 0, 0) 
    return d
    
df_o2 =      get_df('data/Jellyfarm/AKS193_1_FF.xlsx')
df_o2_nf =   get_df('data/Jellyfarm/AKS192_1_NF.xlsx')
df_o2_2_nf = get_df('data/Jellyfarm/AKS189_3_NF.xlsx')
df_o2_2 =    get_df('data/Jellyfarm/AKS193_2_FF.xlsx')

def int_and_plot(axis,xx,yy,col,col_line,addlabel):

    f = interpolate.interp1d(yy,xx, assume_sorted = False)
    ynew = np.arange(np.min(yy),np.max(yy),0.3)
    xnew = f(ynew) 
    if addlabel == True:   
        if col == colr_base_field:
            label  = '300-1000m \nfrom farm'
        else: 
            label = '50m \nfrom farm'
        axis.plot(xnew,ynew, zorder =5, linestyle= '--',
            c = col_line,linewidth = ln_field,
            alpha = 0.3)
        sc = axis.scatter(xnew,ynew, zorder = 10, s = 36, 
                            c = col,edgecolor = 'k',
                            alpha = 1,label = label)    
        return sc 
    else:
        axis.plot(xnew,ynew, zorder =5, linestyle= '--',
            c = col_line,linewidth = ln_field,
            alpha = 0.3) 
        sc = axis.scatter(xnew,ynew, zorder = 10, s = 36, 
                            c = col,edgecolor = 'k',
                            alpha = 1)    


# plot Observations data water  
def plot_water(var,axis,title):

    if title == r'$ DIC\ \mu  M $' or title == r'$ Alk\ \mu  M $' :
        axis.scatter(var,tic_depth,color = colr_base_field,
                     zorder=10,edgecolors='k')
        axis.plot(var,tic_depth,'--',color ='k',alpha = 0.3,linewidth = ln_field)                                            
    else: 
        d = df[['Nst/',var,'Pressure']]
        groups = d.groupby('Nst/')

        for n,group in groups: 
            v = group[var].values 
            p = group.Pressure.values 

            if n == 7:
                axis.scatter(v,p, 
                        color = colr_farm, linewidth = 0.7,
                        zorder=10,edgecolors='k')                 
            else:
                axis.scatter(v,p, 
                        color = colr_base_field, linewidth = 0.7,
                        zorder=10,edgecolors='k') 
                axis.plot(v,p, '--',color = 'k', alpha = 0.3,

            linewidth = ln_field)                      
 
# plot Observations data sediments  
def plot_sed(var,var_depth,axis,addlabel): 

    if var != None :
        if addlabel == True:    
            axis.scatter(var,var_depth,color = colr_base_field,linewidth = 0.7,
            edgecolors ='k',zorder=10,label = 'Observartions \nbaseline')      
        else:  
            axis.scatter(var,var_depth,color = colr_base_field,linewidth = 0.7,
            edgecolors ='k',zorder=10)    

        axis.plot(var,var_depth,'--',color = 'k', alpha = 0.3,
        linewidth = ln_field)

# plot model data water          
def plot_brom(var_brom,axis,is_sed): 
    df = df_brom.sel(time=slice('2015-07-01', '2015-10-01'))     
    df = df.where(df.time.dt.dayofyear % 10 == 0,drop = True)
    df['i_norm'] = df.i - df.i[19]    

    if is_sed == True:
        df['z_sed'] = (df.z - df.z[15])*100            
        df = df.where(df.z_sed > -10, drop = True)    
        sed = df.z_sed.values

    elif is_sed == False:  
        df = df.where(df.z < 300., drop = True)    
      
    v_farm = df.where(df['i_norm'] >= -50. , drop = True)
    v_farm = v_farm.where(v_farm['i_norm'] <= 50., drop = True)
    v_farm = v_farm[var_brom]

    v_mid = df.where(df['i_norm'] >= 75. , drop = True)
    v_mid = v_mid.where(v_mid['i_norm'] <= 275., drop = True)
    v_mid = v_mid[var_brom]

    v_base = df.where(df['i_norm'] >= 300., drop = True)
    v_base = v_base[var_brom]
  
    for i in v_mid['i'].values:
        for t in v_mid['time'].values:
            v = v_mid.sel(time= t,i = i)
            if is_sed == False:
                axis.plot(v,v.z.values,'-',color = colr_mid, alpha = a,linewidth = ln)
            elif is_sed == True:
                axis.plot(v,sed,'-',color = colr_mid, alpha = a,linewidth = ln)

    '''for i in v_mid2['i'].values:
        for t in v_mid2['time'].values:
            v = v_mid2.sel(time= t,i = i)
            if is_sed == False:
                axis.plot(v,v.z.values,'-',color = colr_mid2, alpha = a,linewidth = ln)
            elif is_sed == True:
                axis.plot(v,sed,'-',color = colr_mid2, alpha = a,linewidth = ln)'''
               
    for i in v_base['i'].values:
        for t in v_base['time'].values:
            v = v_base.sel(time= t,i = i)
            if is_sed == False:
                axis.plot(v,v.z,'-',color = colr_base_field, alpha = a,linewidth = ln)
            elif is_sed == True:
                axis.plot(v,sed,'-',color = colr_base_field, alpha = a,linewidth = ln)

    for i in v_farm['i'].values:
        for t in v_farm['time'].values:
            v = v_farm.sel(time= t,i = i)
            if is_sed == False:
                axis.plot(v,v.z.values,'-',color = colr_farm, alpha = a,linewidth = ln)
            elif is_sed == True:
                axis.plot(v,sed,'-',color = colr_farm, alpha = a,linewidth = ln)

    if is_sed == True:
        v_fa = v_farm.sel(time = v_farm['time'].values[0],i = v_farm.i.values[0])
        v_ba = v_base.sel(time = v_base['time'].values[0],i = v_base.i.values[0])
        v_mi = v_mid.sel(time = v_mid['time'].values[0],i = v_mid.i.values[0])
        #v_mi2 = v_mid2.sel(time = v_mid2['time'].values[0],i = v_mid2.i.values[0])        
        l1, = axis.plot(v_fa,sed,'-', color = colr_farm,alpha = 1,linewidth = 2,label = '0-50m \nfrom farm')
        l2, = axis.plot(v_mi,sed,'-', color = colr_mid, alpha = 1,linewidth = 2,label = '75-275m \nfrom farm')  
        #l3, = axis.plot(v_mi2,sed,'-', color = colr_mid2, alpha = 1,linewidth = 2,label = '125-200m \nfrom farm')                        
        l3, = axis.plot(v_ba,sed,'-', color = colr_base_field,alpha = 1,linewidth = 2,label = '300-500m \nfrom farm')

          
        return l1,l2,l3 #,l4
fig = plt.figure(figsize=(11.69, 8.27), dpi=100) 
fig.set_facecolor('white') # specify background color

gs = gridspec.GridSpec(2, 6,height_ratios= [4,3])

gs.update(left = 0.07,right = 0.99, 
          bottom = 0.02, top = 0.90,
          wspace = 0.37, hspace= 0.1)

def sbplt(pos): 
    return fig.add_subplot(pos)

ax,ax1  =   sbplt(gs[0,0]), sbplt(gs[1,0])
ax2,ax3 =   sbplt(gs[0,1]), sbplt(gs[1,1]) 
ax4,ax5 =   sbplt(gs[0,2]), sbplt(gs[1,2]) 
ax6,ax7 =   sbplt(gs[0,3]), sbplt(gs[1,3]) 
ax8,ax9 =   sbplt(gs[0,4]), sbplt(gs[1,4])
ax10,ax11 = sbplt(gs[0,5]), sbplt(gs[1,5])

ax.set_ylabel('Depth, m')
ax1.set_ylabel('Depth, cm')

letters = ['(A)','(B)','(C)','(D)','(E)','(F)']
axes = (ax,ax2,ax4,ax6,ax8,ax10)

[axes[n].text(-0.1, 1.1,letters[n], transform=axes[n].transAxes , 
            size = bigger_size) for n in range(0,len(letters))]

# plot figure with observations and model for water and sediment     
def plot_all(var_brom,var_water,var_sed1,var_sed2,title,axis,axis1):    


    water_ticks = np.arange(0, 301,50)                                             
    swi_ticks = np.arange(-3, 8, 1.5)   
    axis.set_yticks(water_ticks)     
    axis1.set_yticks(swi_ticks)     
    axis.xaxis.tick_top()
    axis1.xaxis.tick_top()
    axis.set_ylim(300,0)
    axis1.set_ylim(5.5,-3.1)

    axis.axhspan(310, 0,color='#cce6f5',
                 alpha = 0.4)     
    axis1.axhspan(0, -5,color='#cce6f5',
                 alpha = 0.4)    
    axis1.axhspan(10, 0,color='#dcc196',
                 alpha = 0.4)        
     
    plot_brom(var_brom,axis, is_sed = False)
    l1,l2,l3 = plot_brom(var_brom,axis1, is_sed = True)    #,l4     

    plot_water(var_water,axis,title)
    plot_sed(var_sed1,depth200,axis1,addlabel = True)
    plot_sed(var_sed2,depth199,axis1,addlabel = False)
    axis.set_title(title, y=1.08)
    ax11.legend(title = 'Model data',handles=[l1,l2,l3]) #,l4



plot_all('O2','O2uM',None,None,r'$ O_2\ \mu  M $',ax,ax1) 
plot_all('PO4','PO4uM',po4199,po4200,r'$ PO_4\ \mu  M $',ax2,ax3)
plot_all('NO3','NO3uM',no3199,no3200,r'$ NO_3\ \mu  M $',ax4,ax5)
plot_all('NH4','NH4uM',nh4199,nh4200,r'$ NH_4\ \mu  M $',ax6,ax7)
plot_all('Alk',alk,alk199,alk200,r'$ Alk\ \mu  M $',ax8,ax9)  
plot_all('DIC',tic,tic199,tic200,r'$ DIC\ \mu  M $',ax10,ax11)

sc1 = int_and_plot(ax1,df_o2_2.o2,df_o2_2.depth,colr_farm,'k',addlabel = True)    
int_and_plot(ax1,df_o2.o2,df_o2.depth,colr_farm,'k',addlabel = False) 

sc2 = int_and_plot(ax1,df_o2_nf.o2,df_o2_nf.depth,colr_base_field,'k',addlabel = True)
int_and_plot(ax1,df_o2_2_nf.o2,df_o2_2_nf.depth,colr_base_field,'k',addlabel = False)
ax1.legend(title = 'Field data', handles=[sc1,sc2])


script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/')    

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

if save == True:    
    plt.savefig(results_dir+'Figure5'+'.png',
               facecolor=fig.get_facecolor(),
                edgecolor='none')
else: 
    plt.show()