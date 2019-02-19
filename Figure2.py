'''
Created on 19. july 2017

@author: ELP
'''
import os,sys
print (sys.version) 

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import tkinter as tk 
#from tkinter.filedialog import askopenfilename 
import seaborn as sns
import pandas as pd
from scipy import interpolate

sns.set()
#root = tk.Tk()
#root.withdraw()

fname = (
    r'E:\Users\EYA\Hardnew\data_Hard\BROM_Hardangerfjord_out_1X.nc') 
        #
    #BROM_Hardangerfjord_out_2X.nc 
    #BROM_Hardangerfjord_out_3X.nc 
    #BROM_Hardangerfjord_out_5X.nc 
    #BROM_Hardangerfjord_out_10X.nc 


fh =  Dataset(fname) 
depth_brom = np.array(fh.variables['z'][:])

sed = np.array(fh.variables['z'][15]) # depth of the SWI
n_sed = 12
sed_depth_brom = (np.array(fh.variables['z'][n_sed:])-sed)*100
pomr_brom = np.array(fh.variables['POMR'][:,n_sed:,:])
o2_brom = np.array(fh.variables['O2'][:,n_sed:,:])
time_brom = np.array(fh.variables["time"][:])
i_brom = np.array(fh.variables['i'][:])

len_time = len(time_brom) 
len_i = len(i_brom)

col_farm = '#842967'    
col_farm_dark = '#571b44'
col_base = '#5a773d'
col_base_dark = '#122212'
#c_base_line = '#122212'
step = int(len_time/10)

path = 'data/Jellyfarm/SedCharacteristics_Hardangerfj_BCSamples_eya.xlsx'

depth_dict = {'0-1':0.5,'1-2':1.5,'2-5':3.5}
df = pd.read_excel(path,skiprows = 5,
                   usecols=(3,4,16,21),
                   names = ('category','depth_str','n_FF','n_NF'))
df = df[:-1]
df.category = df.category.fillna(method='ffill')
df.depth_str = df.depth_str.fillna(method='ffill')   

depth = []
for n in df.depth_str:
    d = depth_dict[n] 
    depth.append(d)     
df['depth'] = depth 

df_mean = df.groupby('depth').mean()
                 
def plot_pomr(var_brom,title):  
    ax.set_title(title)    
    for day in range(0,len_time,step):          
        for n in range(0,len_i): 
            if n == 19: # 19 is farm
                c = col_farm
                al = 1
                l = 1
            else: 
                c = col_base
                al = 0.5    
                l = 0.5
            ax.plot(var_brom[day,:,n],sed_depth_brom,
                      color = c,alpha = al,linewidth = l,zorder = 1) 
            
    ax.scatter(df['n_FF'],df.depth,s = 35,
                 zorder = 10,c = col_farm,edgecolor = 'k',alpha = 1,label = 'field "Near Farm"')
    ax.scatter(df['n_NF'],
                 df.depth,zorder = 10,s = 35,
                 c = col_base,alpha = 0.7,edgecolor = 'k',label = 'field "Not Farm"')  
      
    ax.scatter(df_mean['n_FF'],df_mean.index,c = col_farm,
               edgecolor = 'k',label = 'field "Near Farm" \nmean',
              zorder = 10,s = 36) 
           
    ax.scatter(df_mean['n_NF'],df_mean.index,edgecolor = 'k', c = col_base, 
              s = 36,zorder = 10,label = 'field "Not Farm" \nmean') 
       
    ax.plot(df_mean['n_FF'],df_mean.index,c = col_farm_dark,label = 'field "Near Farm" \nmean',
              zorder = 5,linestyle = '--') 
           
    ax.plot(df_mean['n_NF'],df_mean.index, c = col_base_dark,linestyle = '--',
              zorder = 5,label = 'field "Not Farm" \nmean')
    
            
def plot_o2(var_brom,title,axis):  
    axis.set_title(title)    
    for day in range(0,len_time,step):          
        for n in range(0,len_i,1): 
            c = col_base
            al = 0.1     
            axis.plot(var_brom[day,:,n],sed_depth_brom,
                      color = c,alpha = al,linewidth = 0.5,zorder = 1)         
        axis.plot(var_brom[day,:,19],sed_depth_brom,
                color = col_farm,alpha = 1,linewidth = 2,zorder = 2,label = 'Model Data') 
         

                                          
fig1 = plt.figure(figsize=(7.69, 4.27), dpi=100) 
fig1.set_facecolor('white') 
gs = gridspec.GridSpec(1, 2) 
gs.update(left = 0.1,right = 0.97, 
          bottom = 0.07, top = 0.95,
          wspace = 0.35, hspace= 0.25)
ax = fig1.add_subplot(gs[0]) 
ax1 = fig1.add_subplot(gs[1])
axes = (ax,ax1)
col_water = '#cce6f5'
col_sed = '#dcc196'
for a in axes:
    a.axhspan(0,-10,color= col_water,
              alpha = 0.4 )
    a.axhspan(10,0,color= col_sed, 
              alpha = 0.4)
    a.set_ylabel('Depth, cm')
    a.set_ylim(4,-8)
    

plot_pomr(pomr_brom,r'$ POMR\ \mu  M $')


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


def int_and_plot(xx,yy,col,col_line):

    f = interpolate.interp1d(yy,xx, assume_sorted = False)
    ynew = np.arange(np.min(yy),np.max(yy),0.3)
    xnew = f(ynew) 
    
    plot_o2(o2_brom,r'$ O_2\  \mu  M $',ax1) 
    ax1.plot(xnew,ynew, zorder =5, linestyle= '--',# s = 45,
                     c = col_line,
                     alpha = 1,label = 'field "Near Farm"')
    ax1.scatter(xnew,ynew, zorder = 10, s = 36, 
                     c = col,edgecolor = 'k',
                     alpha = 1,label = 'field "Near Farm"')    
    
int_and_plot(df_o2_2.o2,df_o2_2.depth,col_farm,col_farm_dark)    
int_and_plot(df_o2.o2,df_o2.depth,col_farm,col_farm_dark) 
int_and_plot(df_o2_nf.o2,df_o2_nf.depth,col_base,col_base_dark) 
int_and_plot(df_o2_2_nf.o2,df_o2_2_nf.depth,col_base,col_base_dark) 

#ax.legend() 
plt.show()
#plt.savefig(results_dir+'Figure2'+'.png', #'.eps'
#           facecolor=fig1.get_facecolor(),
#            edgecolor='none')   