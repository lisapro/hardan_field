'''
Scrip to plot Figure 6 
Time-Series Profle
Baseline vs Farm 1x 
'''
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np 
import xarray as xr 
import util
import numpy.ma as ma
from matplotlib import ticker
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
sns.set_style("whitegrid")
register_matplotlib_converters()


df_brom = xr.open_dataset(util.path_brom1)

def get_z(index,col):
    return df_brom[index][util.start_day:util.stop_day,:,col].values

def get_levels(arr1,arr2):
    #nlev = utl200
    vmin = np.min((arr1,arr2))
    vmax =  np.max((arr1,arr2))
    return np.linspace(vmin,vmax,util.nlev)


def plot_param(param,axis,axis1,axis2,axis3,axis_cb,axis_cb_sed):

    
    #sns.palplot(sns.color_palette("cubehelix", 8)) 

    z_baseline = get_z(param,util.baseline_col).T
    if param == 'Phy':
        z_farm = get_z(param,16).T
    else:     
        z_farm = get_z(param,util.farm_col).T
    #z_farm = get_z(param,util.farm_col).T

    
    x = df_brom.time[util.start_day:util.stop_day].values
    y = df_brom.z.values
    sed = 15 
    y_sed = ((y - y[sed])*100)   
    sed2 = 13
    

    sed_levels = get_levels(z_baseline[sed2:,:],z_farm[sed2:,:]) 
    if param == 'Phy': 
        levels = np.linspace(0,7,util.nlev)   
    else:    
        levels = get_levels(z_baseline[:sed2,:],z_farm[:sed2,:]) 
    #vmin = np.min((np.min(z_baseline[:sed2,:]),np.min(z_farm[:sed2,:])))
    #vmax = np.max((z_baseline[:sed2,:],z_farm[:sed2,:]))
    #levels = np.linspace(vmin,vmax,nlev)

    X,Y = np.meshgrid(x,y[:sed2])  
    X_sed,Y_sed = np.meshgrid(x,y_sed[sed2:]) 
    cmap = plt.get_cmap('gist_earth') #jet') ##sns.cubehelix_palette(n_colors = 1,as_cmap=True)
    # print (param,vmin,vmax)
    CS_base = axis.contourf(X,Y, z_baseline[:sed2,:], levels = levels,extend="both",cmap = cmap) #
    CS_farm = axis1.contourf(X,Y, z_farm[:sed2,:], levels = levels,cmap = cmap) #extend="both",
    axis1.xaxis.set_major_locator(ticker.AutoLocator())
    CS_base_sed = axis2.contourf(X_sed,Y_sed, z_baseline[sed2:,:], levels = sed_levels, extend="both",cmap = cmap)
    CS_farm_sed = axis3.contourf(X_sed,Y_sed, z_farm[sed2:,:], levels = sed_levels,extend="both",cmap = cmap)


    tick_locator = ticker.MaxNLocator(nbins=4)

    cb = plt.colorbar(CS_base,cax = axis_cb)

    cb.locator = tick_locator
    cb.update_ticks()

    cb_sed = plt.colorbar(CS_base_sed,cax = axis_cb_sed)

    cb_sed.locator = tick_locator
    cb_sed.update_ticks()

    axis.set_xticklabels([])
    axis1.set_xticklabels([])
    axis.set_ylabel('Depth,m')
    
    axis.set_ylim(312,0)
    axis1.set_ylim(312,0)
    axis2.set_ylim(5,-5)
    axis3.set_ylim(5,-5)   
    axis2.set_ylabel('Depth,cm')

    mnths = ['F','M','A','M','J','J','A','S','O','N','D','J']
    axis2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axis3.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axis2.set_xticklabels(mnths)
    axis3.set_xticklabels(mnths)
    axis2.axhline(0,linestyle = '--',linewidth = 0.5,color = 'w')
    axis3.axhline(0,linestyle = '--',linewidth = 0.5,color = 'w') 

    axis.set_yticks([50,150,250])
    axis1.set_yticks([50,150,250])

    axis.tick_params(axis='y', pad = 0.01)
    axis2.tick_params(axis='y', pad = 1)

    #fig.autofmt_xdate()



fig = plt.figure(figsize=(8.27,11), dpi=100) 

gs = gridspec.GridSpec(5, 2,width_ratios = [30,1]) 
gs.update(left = 0.07,right = 0.93, 
          bottom = 0.04, top = 0.95,
          wspace = 0.05, hspace= 0.3)


h = 0.06
w = 0.15

ratios = [3,2]
def create_gs(pos):
    return gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[pos],
                                hspace=h,wspace=w,height_ratios=ratios) 
def call_create_gs_cb(poss):
    return [create_gs_cb(n) for n in poss]

def create_gs_cb(pos):
    return gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[pos],
                                hspace=h,wspace=w,height_ratios=ratios) 

def call_create_gs(poss):
    return [create_gs(n) for n in poss]

gs00,gs01,gs02,gs03,gs04 = call_create_gs([0,2,4,6,8])
gs00_cb,gs01_cb,gs02_cb,gs03_cb,gs04_cb = call_create_gs_cb([1,3,5,7,9])

def sbplt(pos): 
    return fig.add_subplot(pos)

ax,ax1,ax_cb  =   sbplt(gs00[0,0]), sbplt(gs00[0,1]), sbplt(gs00_cb[0])
ax2,ax3,ax_cb_sed =   sbplt(gs00[1,0]), sbplt(gs00[1,1]), sbplt(gs00_cb[1]) 

ax_1,ax1_1,ax1_cb  =   sbplt(gs01[0,0]), sbplt(gs01[0,1]), sbplt(gs01_cb[0])
ax2_1,ax3_1, ax1_cb_sed =   sbplt(gs01[1,0]), sbplt(gs01[1,1]), sbplt(gs01_cb[1])

ax_2,ax1_2,ax2_cb  =   sbplt(gs02[0,0]), sbplt(gs02[0,1]), sbplt(gs02_cb[0])
ax2_2,ax3_2, ax2_cb_sed  =   sbplt(gs02[1,0]), sbplt(gs02[1,1]), sbplt(gs02_cb[1]) 

ax_3,ax1_3,ax3_cb  =   sbplt(gs03[0,0]), sbplt(gs03[0,1]), sbplt(gs03_cb[0])
ax2_3,ax3_3, ax3_cb_sed  =   sbplt(gs03[1,0]), sbplt(gs03[1,1]), sbplt(gs03_cb[1]) 

ax_4,ax1_4,ax4_cb  =   sbplt(gs04[0,0]), sbplt(gs04[0,1]), sbplt(gs04_cb[0])
ax2_4,ax3_4, ax4_cb_sed  =   sbplt(gs04[1,0]), sbplt(gs04[1,1]), sbplt(gs04_cb[1]) 



plot_param('Phy',ax,ax1,ax2,ax3,ax_cb,ax_cb_sed) 
ax.set_title(r'$Phy\ baseline\ \mu M\ N$')
ax1.set_title(r'$Phy\ Farm\ 1x\ \mu M\ N$')

plot_param('POMR',ax_1,ax1_1,ax2_1,ax3_1,ax1_cb,ax1_cb_sed) 
ax_1.set_title(r'$POMR\ baseline\ \mu M\ N$')
ax1_1.set_title(r'$POMR\ Farm\ 1x\ \mu M\ N$')

plot_param('O2',ax_2,ax1_2,ax2_2,ax3_2,ax2_cb,ax2_cb_sed) 
ax_2.set_title(r'$O_2\ baseline\ \mu M\ N$')
ax1_2.set_title(r'$O_2\ Farm\ 1x\ \mu M\ N$')

plot_param('NH4',ax_3,ax1_3,ax2_3,ax3_3,ax3_cb,ax3_cb_sed) 
ax_3.set_title(r'$NH_4\ baseline\ \mu M\ N$')
ax1_3.set_title(r'$NH_4\ Farm\ 1x\ \mu M\ N$')

plot_param('DOMR',ax_4,ax1_4,ax2_4,ax3_4,ax4_cb,ax4_cb_sed)
ax_4.set_title(r'$DOMR\ baseline\ \mu M\ N$')
ax1_4.set_title(r'$DOMR\ Farm\ 1x\ \mu M\ N$')
    
#plt.show()
plt.savefig('Results/Figure6.png')