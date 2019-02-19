'''
Created on 19. july 2017

@author: ELP
'''


import os,sys
import netCDF4
from netCDF4 import Dataset
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import time
import xarray as xr 
from scipy import interpolate


plt.style.use('bmh') 

# -- define fonts
SMALL_SIZE  = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 12 #9
plt.rc('font',    size = MEDIUM_SIZE)   # controls default text sizes
plt.rc('axes', labelsize = BIGGER_SIZE)   # fontsize of the x and y labels

plt.rc('xtick', labelsize = BIGGER_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize = BIGGER_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize = SMALL_SIZE)   # legend fontsize


df = pd.read_csv('data/wat_hardan2.txt')

#Observations
tic = [2190.56, 2021.71, 988.85, 999.57]
tic_depth = [470,200,0, 10]
alk = [2301.98,2180.317, 1059.57, 1058.218]

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


fname = (
     r'E:\Users\EYA\Hardnew\data_Hard\BROM_Hardangerfjord_out_1X.nc')


colr1 = '#ff8800'
colr_base = '#2e3236'
colr_base_field = '#519344'
colr_farm = '#c75641'
baseline_col = 0
farm_col = 19
a = 1
ln  = 0.4
ln_field = 1



df_brom = xr.open_dataset(fname)

depth_brom = df_brom['z'].values
sed = depth_brom[15]
sed_depth_brom  = (depth_brom - sed )*100
time_brom = df_brom['time'].values
len_time = len(time_brom)
i_brom = df_brom['i'].values
len_i = len(i_brom)

# Add oxygen 

def get_df(path):
    d = pd.read_excel(path,skiprows = 1,usecols = [1,4,7],
                      names = ['depth','time','o2'])
    d.depth = d.depth / -10000
    #d = d.resample('30s', on='time').mean()  
    d.o2 = d['o2'].where(d.o2 >= 0, 0) 
    return d
    
df_o2 = get_df('data/Jellyfarm/AKS193_1_FF.xlsx')
df_o2_nf = get_df('data/Jellyfarm/AKS192_1_NF.xlsx')
df_o2_2_nf = get_df('data/Jellyfarm/AKS189_3_NF.xlsx')
df_o2_2 = get_df('data/Jellyfarm/AKS193_2_FF.xlsx')


def int_and_plot(axis,xx,yy,col,col_line):

    f = interpolate.interp1d(yy,xx, assume_sorted = False)
    ynew = np.arange(np.min(yy),np.max(yy),0.3)
    xnew = f(ynew) 
    
    
    axis.plot(xnew,ynew, zorder =5, linestyle= '--',# s = 45,
                     c = col_line,linewidth = ln_field,
                     alpha = 1,label = 'field "Near Farm"')
    axis.scatter(xnew,ynew, zorder = 10, s = 36, 
                     c = col,edgecolor = 'k',
                     alpha = 1,label = 'field "Near Farm"')    





# plot Observations data water  
def plot_water(var,axis,title):
    if title == r'$ DIC\ \mu  M $' :
        axis.scatter(var,tic_depth,color = colr_base_field,
                     zorder=10,edgecolors='k')
    elif title == r'$ Alk\ \mu  M $' :
        axis.scatter(var,tic_depth,color = colr_base_field,
                     zorder=10,edgecolors='k')           
    else: 
        for m in range(0,6):        
            axis.plot(var[start_list[m]:end_list[m]], 
                    df.Pressure[start_list[m]:end_list[m]],
                    'o', color = colr_base_field, markeredgecolor ='k',zorder = 10,  
                    alpha = a)        
            #axis.plot(var[start_list[m]:end_list[m]], 
            #        df.Pressure[start_list[m]:end_list[m]],
            #        '--', color = colr_base_field, zorder = 5, linewidth = ln_field, 
            #        alpha = a)   


# plot Observations data sediments  
def plot_sed(var,var_depth,axis): 
    if var == None :
        pass
    else:    
        axis.plot(var,var_depth,'--',color = colr_base,
        linewidth = ln_field)
        axis.plot(var,var_depth,'o',color = colr_base_field,
        markeredgecolor ='k')

# plot model data water          
def plot_brom(var_brom,axis):    
    v = df_brom[var_brom].values
    for day in range(731,1096,10):  
        axis.plot(v[day,:15,baseline_col],depth_brom[:15],
                      color = colr_base, alpha = a, 
                      linewidth = ln) 
        axis.plot(v[day,:15,farm_col],depth_brom[:15],
                      color = colr_farm, alpha = a, 
                      linewidth = ln) 

# plot model data sediments                             
def plot_brom_sed(var_brom,axis): 
    v = df_brom[var_brom].values 
    for day in range(731,1096,10):  
        axis.plot(v[day,13:,baseline_col],sed_depth_brom[13:],
                      color = colr_base, alpha = a,
                      linewidth = ln,zorder = 1 ) 
        axis.plot(v[day,13:,farm_col],sed_depth_brom[13:],
                      color = colr_farm, alpha = a,
                      linewidth = ln,zorder = 1 ) 

fig1 = plt.figure(figsize=(11.69, 8.27), dpi=100) 
fig1.set_facecolor('white') # specify background color

gs = gridspec.GridSpec(2, 6)

gs.update(left = 0.07,right = 0.99, 
          bottom = 0.07, top = 0.95,
          wspace = 0.3, hspace= 0.15)

ax = fig1.add_subplot(gs[0,0]) 
ax1 = fig1.add_subplot(gs[1,0])

ax2 = fig1.add_subplot(gs[0,1]) 
ax3 = fig1.add_subplot(gs[1,1])

ax4 = fig1.add_subplot(gs[0,2]) 
ax5 = fig1.add_subplot(gs[1,2])

ax6 = fig1.add_subplot(gs[0,3])
ax7 = fig1.add_subplot(gs[1,3]) 

ax8 = fig1.add_subplot(gs[0,4])
ax9 = fig1.add_subplot(gs[1,4])

ax10 = fig1.add_subplot(gs[0,5])
ax11 = fig1.add_subplot(gs[1,5])

ax.set_ylabel('Depth, m')
ax1.set_ylabel('Depth, cm')

letters = ['(A)','(B)','(C)','(D)','(E)','(F)']
axes = (ax,ax2,ax4,ax6,ax8,ax10)

[axes[n].text(-0.1, 1.05,letters[n], transform=axes[n].transAxes , 
            size = BIGGER_SIZE) for n in range(0,len(letters))]
'''
#n = 0
#for axis in (ax1,ax3,ax5,ax7,ax9,ax11): 
#    #axis.yaxis.set_label_coords(-0.1, 0.6)
#    axis.text(-0.2, -0.12, letters[n], transform=axis.transAxes , 
#            size = BIGGER_SIZE) #, weight='bold')      
#    n=n+1
'''

# plot figure with observations and model for water and sediment     
def plot_all(var_brom,var_water,var_sed1,var_sed2,title,axis,axis1):    

    axis.set_ylim(310,0)
    axis1.set_ylim(3.1,-3.1)
    water_ticks = np.arange(0, 310, 100)                                             
    swi_ticks = np.arange(-3, 3, 1.5)   
    axis.set_yticks(water_ticks)     
    axis1.set_yticks(swi_ticks)     
   
    axis.axhspan(310, 0,color='#cce6f5',
                 alpha = 0.4, label = "water")     
    axis1.axhspan(0, -5,color='#cce6f5',
                 alpha = 0.4, label = "water")    
    axis1.axhspan(5, 0,color='#dcc196',
                 alpha = 0.4, label = "sediment")        
     
    plot_brom(var_brom,axis)
    plot_brom_sed(var_brom,axis1)            
    plot_water(var_water,axis,title)
    plot_sed(var_sed1,depth200,axis1)
    plot_sed(var_sed2,depth199,axis1)
    axis.set_title(title)
    

plot_all('O2',df.O2uM,None,None,r'$ O_2\ \mu  M $',ax,ax1) 
plot_all('PO4',df.PO4uM,po4199,po4200,r'$ PO_4\ \mu  M $',ax2,ax3)
plot_all('NO3',df.NO3uM,no3199,no3200,r'$ NO_3\ \mu  M $',ax4,ax5)
plot_all('NH4',df.NH4uM,nh4199,nh4200,r'$ NH_4\ \mu  M $',ax6,ax7)
plot_all('Alk',alk,alk199,alk200,r'$ Alk\ \mu  M $',ax8,ax9)  
plot_all('DIC',tic,tic199,tic200,r'$ DIC\ \mu  M $',ax10,ax11)

int_and_plot(ax1,df_o2_2.o2,df_o2_2.depth,colr_farm,colr_farm)    
int_and_plot(ax1,df_o2.o2,df_o2.depth,colr_farm,colr_farm) 
int_and_plot(ax1,df_o2_nf.o2,df_o2_nf.depth,colr_base_field,colr_base_field) 
int_and_plot(ax1,df_o2_2_nf.o2,df_o2_2_nf.depth,colr_base_field,colr_base_field) 



script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/')    

save = False 
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

if save == True:    
    plt.savefig(results_dir+'Figure1'+'.png', #'.eps'
               facecolor=fig1.get_facecolor(),
                edgecolor='none')
else: 
    plt.show()

