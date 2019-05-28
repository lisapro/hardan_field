'''
Created on 19. july 2017

@author: ELP
'''

import os,sys
#import netCDF4
#from netCDF4 import Dataset
import csv,time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import xarray as xr 
from scipy import interpolate
from util import path_brom1

plt.style.use('bmh') 

r'''    The module plot Figure 5 for the 
        Jellyfarm project, Hardangerfjord case , 2D BROM
        Data from the Brom are plotted from start_day to stop_day,
        For 2 columns: Baseline and "Farm"

Figure5.py 
Для параметров:

O2, H2S, pHmeasured, pCO2, ALk DIC, Si. PO4, Mn diss, Fe diss

Модельный расчет  (одномерная модель, надо брать для одного года любого) и данные (экселевский файл) лежат в 

e:\Users\EYA\Horten\_for_Paper\ '''


start_day = 0
stop_day = 365




# -- define fonts
small_size  = 8
medium_size = 10
bigger_size = 13
plt.rc('font',    size = medium_size)   # controls default text sizes
plt.rc('axes', labelsize = bigger_size)   # fontsize of the x and y labels

plt.rc('xtick', labelsize = bigger_size)   # fontsize of the tick labels
plt.rc('ytick', labelsize = bigger_size)   # fontsize of the tick labels
plt.rc('legend', fontsize = small_size)   # legend fontsize


#df = pd.read_csv('data/wat_hardan2.txt') #HH_180824_chem.xls


colr1 = '#ff8800'
colr_base = '#2e3236'
colr_base_field = '#519344'
colr_farm = '#c75641'

a = 1
ln  = 0.4
ln_field = 1

df_brom = xr.open_dataset(r'e:\Users\EYA\Horten\_for_Paper\BROM_Horten_out.nc')

depth_brom = df_brom['z'].values
nsed = 8
sed_depth_brom  = (depth_brom - depth_brom[nsed])*100

df_field = pd.read_excel(r'e:\Users\EYA\Horten\_for_Paper\HH_180824_chem.xls')
df_field = df_field.drop([0,7,8,9,10])
#print (df_field)
'''# Add oxygen 
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
df_o2_2 =    get_df('data/Jellyfarm/AKS193_2_FF.xlsx')'''
   

# plot Observations data water  
def plot_water(var,axis,title):
    var_dict = {'PO4':'PO4','Si':'Si ','Mn2':'Mn_diss','Fe2':'Fe_diss','pCO2':'pCO2','pH':'pH, NBS','H2S':'H2S','O2':'O2','DIC':'CO2','Alk':'Alk'}
    var = var_dict[var]
    if var == None:
        pass 
    else:
        #print (df_field[var].values,df_field['Pressure'].values)
        axis.plot(df_field[var].values,df_field['Pressure'].values,'o--',alpha = 1,zorder = 10,c = colr_base_field)



# plot model data water          
def plot_brom(var_brom,axis):    
    v = df_brom[var_brom].values
    for day in range(start_day,stop_day,10):  
        axis.plot(v[day,:nsed],depth_brom[:nsed],
                      color = colr_base, alpha = a, 
                      linewidth = ln) 


# plot model data sediments                             
def plot_brom_sed(var_brom,axis): 
    v = df_brom[var_brom].values 
    for day in range(start_day,stop_day,10):  
        axis.plot(v[day,nsed-5:],sed_depth_brom[nsed-5:],
                      color = colr_base, alpha = a,
                      linewidth = ln,zorder = 1 ) 

def make_fig():
    fig = plt.figure(figsize=(11.69, 8.27), dpi=100) 
    fig.set_facecolor('white') # specify background color

    gs = gridspec.GridSpec(2, 5)

    gs.update(left = 0.07,right = 0.99, 
            bottom = 0.05, top = 0.95,
            wspace = 0.37, hspace= 0.1)

    def sbplt(pos): 
        return fig.add_subplot(pos)

    ax,ax1  =   sbplt(gs[0,0]), sbplt(gs[1,0])
    ax2,ax3 =   sbplt(gs[0,1]), sbplt(gs[1,1]) 
    ax4,ax5 =   sbplt(gs[0,2]), sbplt(gs[1,2]) 
    ax6,ax7 =   sbplt(gs[0,3]), sbplt(gs[1,3]) 
    ax8,ax9 =   sbplt(gs[0,4]), sbplt(gs[1,4])
    #ax10,ax11 = sbplt(gs[0,5]), sbplt(gs[1,5])

    ax.set_ylabel('Depth, m')
    ax1.set_ylabel('Depth, cm')

    letters = ['(A)','(B)','(C)','(D)','(E)']
    axes = (ax,ax2,ax4,ax6,ax8)

    [axes[n].text(-0.1, 1.05,letters[n], transform=axes[n].transAxes , 
                size = bigger_size) for n in range(0,len(letters))]
    return (fig,ax,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9)
# plot figure with observations and model for water and sediment     
def plot_all(var_brom,var_water,var_sed1,var_sed2,title,axis,axis1):    
    ymax = depth_brom[nsed]
    axis.set_ylim(ymax,0)
    axis1.set_ylim(3.1,-3.1)
    water_ticks = np.arange(0, ymax, 5)                                             
    swi_ticks = np.arange(-3, 3, 1.5)   
    axis.set_yticks(water_ticks)     
    axis1.set_yticks(swi_ticks)     
   
    axis.axhspan(30, 0,color='#cce6f5',
                 alpha = 0.4, label = "water")     
    axis1.axhspan(0, -5,color='#cce6f5',
                 alpha = 0.4, label = "water")    
    axis1.axhspan(5, 0,color='#dcc196',
                 alpha = 0.4, label = "sediment")        
     
    plot_brom(var_brom,axis)
    plot_brom_sed(var_brom,axis1)            
    plot_water(var_brom,axis,title)
    #plot_sed(var_sed1,depth200,axis1)
    #plot_sed(var_sed2,depth199,axis1)
    axis.set_title(title)

def plot1(save):    
    #H2S, pHmeasured, pCO2, ALk DIC, 
    fig,ax,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9  = make_fig()  
    #plot_all('O2',None,None,None,r'$ O_2\ \mu  M $',ax,ax1) 
    plot_all('DIC',None,None,None,r'$ DIC\ \mu  M $',ax,ax1)    
    plot_all('H2S',None,None,None,r'$ H_2S\ \mu  M $',ax2,ax3)
    plot_all('pH',None,None,None,r'$ pH $',ax4,ax5)
    plot_all('pCO2',None,None,None,r'$ pCO_2\ $',ax6,ax7)
    plot_all('Alk',None,None,None,r'$ Alk\ \mu  M $',ax8,ax9)  #

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/')    

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    if save == True:    
        plt.savefig(results_dir+'Figure5_2_1'+'.png',
                facecolor=fig.get_facecolor(),
                    edgecolor='none')
    else: 
        plt.show()
def plot2(save):      
    #O2, Si. PO4, Mn diss, Fe diss    
    fig,ax,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9  = make_fig()
    plot_all('O2',None,None,None,r'$ O_2\ \mu  M $',ax,ax1) 
    plot_all('Si',None,None,None,r'$ Si\ \mu  M $',ax2,ax3)
    plot_all('PO4',None,None,None,r'$PO_4$',ax4,ax5)
    plot_all('Mn2',None,None,None,r'$Mn II\ $',ax6,ax7)
    plot_all('Fe2',None,None,None,r'$Fe II\ $',ax8,ax9)  #

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'Results/')    

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    if save == True:    
        plt.savefig(results_dir+'Figure5_2_2'+'.png',
                facecolor=fig.get_facecolor(),
                    edgecolor='none')
    else: 
        plt.show()

plot1(save = True)
make_fig()
plot2(save = True)