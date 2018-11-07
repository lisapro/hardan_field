'''
Created on 19. july 2017

@author: ELP
'''

import os,sys
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tkinter as tk 
from tkinter.filedialog import askopenfilename 
import seaborn as sns
import pandas as pd
sns.set()
root = tk.Tk()
root.withdraw()

fname = (
    r'E:\Users\EYA\Hardnew\data_Hard\BROM_Hardangerfjord_out.nc')

fh =  Dataset(fname) 
depth_brom = np.array(fh.variables['z'][:])

sed = np.array(fh.variables['z'][15]) # depth of the SWI
n_sed = 13
sed_depth_brom = (np.array(fh.variables['z'][n_sed:])-sed)*100
domr_brom = np.array(fh.variables['DOMR'][:,n_sed:,:])
pomr_brom = np.array(fh.variables['POMR'][:,n_sed:,:])

time_brom = np.array(fh.variables["time"][:])
i_brom = np.array(fh.variables['i'][:])

len_time = len(time_brom) 
len_i = len(i_brom)

col_farm = 'k'    
col_base = 'r'

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

    
                         
def plot_farm(var_brom,title,axis):  
    axis.set_title(title)    
    for day in range(0,len_time,step):          
        for n in range(0,len_i): 
            if n == 19: # 19 is farm
                c = col_farm
                al = 1
            else: 
                c = col_base
                al = 0.2     
            axis.plot(var_brom[day,:,n],sed_depth_brom,
                      color = c,alpha = al,linewidth = 0.5) 

    axis.scatter(df['n_FF'],df.depth,
                 zorder = 10,c = 'k',alpha = al)
    axis.plot(df_mean['n_NF'],df_mean.index,
              'ko--',zorder = 10)    
def plot_base(var_brom,title,axis):  
    axis.set_title(title) 
    al,l = 0.2,0.5  
    
    for day in range(0,len_time,step):  
        for n in range(0,5):
            axis.plot(var_brom[day,:,n],sed_depth_brom,
                      color = col_base,alpha = al,
                      linewidth = l)             
            
        for n in range(35,40):
            axis.plot(var_brom[day,:,n],sed_depth_brom,
                      color = col_base,alpha = al,
                      linewidth = l) 
            
    axis.scatter(df['n_NF'],
                 df.depth,zorder = 10,
                 c = 'k',alpha = al)
    axis.plot(df_mean['n_NF'],df_mean.index,
              'ko--',zorder = 10)
                                          
fig1 = plt.figure(figsize=(10.69, 7.27), dpi=100) 
fig1.set_facecolor('white') 

gs = gridspec.GridSpec(2, 2)
gs.update(left = 0.07,right = 0.97, 
          bottom = 0.07, top = 0.95,
          wspace = 0.25, hspace= 0.25)

ax = fig1.add_subplot(gs[0,0]) 
ax1 = fig1.add_subplot(gs[0,1])
ax2 = fig1.add_subplot(gs[1,0]) 
ax3 = fig1.add_subplot(gs[1,1])

axes = (ax,ax1,ax2,ax3)
for a in axes:
    a.axhspan(0,-10,color='#cce6f5',
              alpha = 0.4, label = "water")
    a.axhspan(10,0,color='#dcc196', 
              alpha = 0.4, label = "sediment")
    a.set_ylabel('Depth, cm')
    a.set_ylim(7.5,-10)
    
plot_farm(pomr_brom,r'$ POMR\ Farm\ \mu  M $',ax) 
plot_base(pomr_brom,r'$ POMR\ Baseline \ \mu  M $',ax1) 

plot_farm(domr_brom,r'$ DOMR\ Farm\ \mu  M $',ax2) 
plot_base(domr_brom,r'$ DOMR\ Baseline \ \mu  M $',ax3) 



script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/')    
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
    
plt.show()
