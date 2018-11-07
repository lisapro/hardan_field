'''
Created on 19. july 2017

@author: ELP
'''

# ---------- LOAD LIBRARIES ------------  

import os,sys
import netCDF4
from netCDF4 import Dataset
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tkinter as tk 
from tkinter.filedialog import askopenfilename 
root = tk.Tk()
root.withdraw()
import time

# ---------- DEFINE PLOT STYLE ------------  

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

# ---------- READ FILEs WITH OBSERVARION DATA ------------  

with open('data/wat_hardan2.txt', 'r') as f:
    # important to specify delimiter right 
    reader = csv.reader(f, delimiter=',')
    r = []
    for row in reader:
        r.append(row)        
r1 = np.transpose(np.array(r[1:])) 

st,nh4  = r1[0], r1[1]    
po4 = r1[2]
ptot = r1[3]
ntot = r1[4]
si = r1[5]  
no3 = r1[6]  
o2 = r1[7]  
doc = r1[8]  
pressure = r1[9]  
fe3 = r1[10]  
mn2 = r1[11]  
dop = r1[12]  
don = r1[13] 

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

# ---------- LOAD FILE WITH MODEL DATA (NetCDF) ------------  

fname = (
    r'E:\Users\EYA\Hardnew\data_Hard\BROM_Hardangerfjord_out.nc')

fh =  Dataset(fname) 

kz =  np.array(fh.variables['Kz'][:,:,0])
depth_brom = np.array(fh.variables['z'][:])
sed = np.array(fh.variables['z'][15]) # depth of the SWI
sed_depth_brom = np.array(fh.variables['z'][:])-sed
sed_depth_brom = sed_depth_brom *100
si_brom = np.array(fh.variables['Si'][:,:,:]) 
nh4_brom = np.array(fh.variables['NH4'][:,:,:])
po4_brom = np.array(fh.variables['PO4'][:,:,:])
no3_brom = np.array(fh.variables['NO3'][:,:,:])
o2_brom = np.array(fh.variables['O2'][:,:,:])
alk_brom = np.array(fh.variables['Alk'][:,:,:])
tic_brom = np.array(fh.variables['DIC'][:,:,:])
time_brom = np.array(fh.variables["time"][:])

len_time = len(time_brom)
i_brom = np.array(fh.variables['i'][:]) 
len_i = len(i_brom)


# ---------- SUBROUTINES TO PLOT ------------ 

# plot Observations data water  
def plot_water(var,axis,title):
    if title == r'$ DIC\ \mu  M $' : # 'DIC': #var == alk or var == tic: 
        axis.scatter(var,tic_depth,color = '#825572',
                     zorder=10,edgecolors='k')
    elif title == r'$ Alk\ \mu  M $' :
        axis.scatter(var,tic_depth,color = '#825572',
                     zorder=10,edgecolors='k')           
    else: 
        for m in range(0,6):        
            axis.plot(var[start_list[m]:end_list[m]], 
                    pressure[start_list[m]:end_list[m]],
                    'o-', color = '#9f6b8c',
                    alpha = 0.7)        
                     
# plot Observations data sediments  
def plot_sed(var,var_depth,axis): 
    if var == None :
        pass
    else:    
        axis.plot(var,var_depth,'o-')

# plot model data water          
def plot_brom(var_brom,axis):    
    for day in range(0,1370,30):  
        # plot every 10'th day  to day 1350 (can be "len_time")
        if  day>1290 and day<1370 :
            col = '#ff8800'
        else :
            col = '#a1a7af'
            
        for n in range(0,10,2): 
            #plot every 1st column in 10 column to avoid zone affected by FF
#       for n in range(0,len_i,2): # plot every 2d column
                #for m in range(0,100):# time_brom-1:
            axis.plot(var_brom[day,:15,n],depth_brom[:15],
                      color =col,alpha = 0.6, 
                      linewidth = 0.4) 

# plot model data sediments                             
def plot_brom_sed(var_brom,axis):  
    for day in range(0,1370,20):  
        if  day>1290 and day<1370 :
            col = '#ff8800'
        else :
            col = '#a1a7af'
        for n in range(0,len_i,2):
            axis.plot(var_brom[day,13:,n],sed_depth_brom[13:],
                      color = col,alpha = 0.3,
                      linewidth = 0.3,zorder = 1 ) 
                              
fig1 = plt.figure(figsize=(11.69, 8.27), dpi=140) 
fig1.set_facecolor('white') # specify background color

gs = gridspec.GridSpec(2, 6)

gs.update(left = 0.07,right = 0.97, 
          bottom = 0.07, top = 0.95,
          wspace = 0.25, hspace= 0.15)


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
''''n = 0
for axis in (ax1,ax3,ax5,ax7,ax9,ax11): 
    #axis.yaxis.set_label_coords(-0.1, 0.6)
    axis.text(-0.2, -0.12, letters[n], transform=axis.transAxes , 
            size = BIGGER_SIZE) #, weight='bold')      
    n=n+1'''


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
     
    #plot_brom(var_brom,axis)
    #plot_brom_sed(var_brom,axis1)            
    plot_water(var_water,axis,title)
    plot_sed(var_sed1,depth200,axis1)
    plot_sed(var_sed2,depth199,axis1)
    axis.set_title(title)
    
# ------------- FINALLY PLOT ! ------------

plot_all(o2_brom,o2,None,None,r'$ O_2\ \mu  M $',ax,ax1) 
plot_all(po4_brom,po4,po4199,po4200,r'$ PO_4\ \mu  M $',ax2,ax3)
plot_all(no3_brom,no3,no3199,no3200,r'$ NO_3\ \mu  M $',ax4,ax5)
plot_all(nh4_brom,nh4,nh4199,nh4200,r'$ NH_4\ \mu  M $',ax6,ax7)
plot_all(alk_brom,alk,alk199,alk200,r'$ Alk\ \mu  M $',ax8,ax9)  
plot_all(tic_brom,tic,tic199,tic200,r'$ DIC\ \mu  M $',ax10,ax11)


script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/')    
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
    
#plt.savefig(results_dir+'Figure5'+'.eps',#png'
#           facecolor=fig1.get_facecolor(),
#            edgecolor='none')
plt.show()
#plt.clf()
#print ('ferdig!')
