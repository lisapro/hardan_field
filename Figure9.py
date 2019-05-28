import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
'''
Scrip to plot Figure 9
Distance Profle
For Farm 3x ,Farm 5x ...
'''

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

h = 0.14
w = 0.05
sed2 = 12

def make_fig_gs(nrows,ncols):
    global fig
    fig = plt.figure(figsize=(8.27,11*nrows/5), dpi=100) 

    gs = gridspec.GridSpec(nrows, ncols) 
    gs.update(left = 0.07,right = 0.9, 
            bottom = 0.08, top = 0.95,
            wspace = 0.39, hspace= 0.42)
    return gs

    
def create_gs(pos):
    return gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[pos],
                                            hspace=h,wspace=w,height_ratios=[3,2],
                                            width_ratios = [15,1]) 
   
def call_create_gs(poss):
    return [create_gs(n) for n in poss]

def sbplt_cb(to_gs): 
    return [fig.add_subplot(a) for a in [to_gs[0,0],to_gs[0,1],to_gs[1,0],to_gs[1,1]]]

def get_df(my_path):
    return xr.open_dataset(my_path)

def get_z(df_brom,param):
    return df_brom[param][util.dist_day,:,:].values.T


def get_levels_fig8(param):
    z5 = get_z(df_brom_5,param)
    z3 = get_z(df_brom_3,param)
    sed_levels = util.get_levels(z5[:,sed2:],z3[:,sed2:]) 
    levels = util.get_levels(z5[:,:sed2],z3[:,:sed2])

    return levels,sed_levels

def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \cdot 10^{{{}}}$'.format(a, b)

def plot_param(param,z,axis,axis_cb,axis_sed,axis_cb_sed):
    if param == 'H2S':
        levels, sed_levels = get_levels_fig8(param)
    elif param == 'Mn2': 
        #z5 = get_z(df_brom_5,param)
        #z3 = get_z(df_brom_3,param)
        sed_levels = np.linspace(0,150,util.nlev) 
        levels = np.linspace(0,150,util.nlev)        
    elif param == 'Fe2':
        levels = np.linspace(0,1.e-4,util.nlev)  
        sed_levels = np.linspace(0,240,util.nlev) 

    X,Y = np.meshgrid(x,y[:sed2])  
    X_sed,Y_sed = np.meshgrid(x,y_sed[sed2:]) 
    cmap = plt.get_cmap('jet') 

    CS_1 = axis.contourf(X,Y, z[:,:sed2].T, extend="both",cmap = cmap, levels = levels)
    CS_1_sed = axis_sed.contourf(X_sed,Y_sed, z[:,sed2:].T, extend="both", cmap = cmap, levels = sed_levels)

    tick_locator = ticker.MaxNLocator(nbins=4)
    axis_cb.set_aspect(4)

    cb = plt.colorbar(CS_1,cax = axis_cb,format=ticker.FuncFormatter(fmt))

    cb.locator = tick_locator
    cb.update_ticks()

    cb_sed = plt.colorbar(CS_1_sed,cax = axis_cb_sed,format=ticker.FuncFormatter(fmt))
    axis_cb_sed.set_aspect(3)
    
    cb_sed.locator = tick_locator
    cb_sed.update_ticks()

    axis.set_xticklabels([])


    axis.set_ylim(307,0)
    axis_sed.set_ylim(5,-5)

    axis_sed.axhline(0,linestyle = '--',linewidth = 0.5,color = 'w')

    axis.set_yticks([50,150,250])

    axis.tick_params(axis='y', pad = 0.01)
    axis_sed.tick_params(axis='y', pad = 1)
    axis_sed.set_xlabel('Distance, m')

# Figure 7 for Farm 1x 
def fig9():
    global gs,x,y,y_sed
    global df_brom_5,df_brom_3
    gs = make_fig_gs(3,2)

    
    df_brom_5 = get_df(util.path_brom5)
    df_brom_3 = get_df(util.path_brom3)   
    

    gs00,gs00_1,gs01,gs01_1,gs02,gs02_1 = call_create_gs([0,1,2,3,4,5])
 
    ax00,ax00_cb, ax00_sed,ax00_sed_cb = sbplt_cb(gs00)  
    ax00_1,ax00_1_cb, ax00_1_sed,ax00_1_sed_cb = sbplt_cb(gs00_1)

    ax01,ax01_cb, ax01_sed,ax01_sed_cb = sbplt_cb(gs01)  
    ax01_1,ax01_1_cb, ax01_1_sed,ax01_1_sed_cb= sbplt_cb(gs01_1)

    ax02,ax02_cb, ax02_sed,ax02_sed_cb = sbplt_cb(gs02)  
    ax02_1,ax02_1_cb, ax02_1_sed,ax02_1_sed_cb= sbplt_cb(gs02_1)

    x = df_brom_3.i.values
    x = x - x[19] # normalize dist by middle column
    y = df_brom_3.z.values
    sed = 15 
    y_sed = ((y - y[sed])*100)   


    plot_param('Mn2',get_z(df_brom_3,'Mn2'),ax00,ax00_cb,ax00_sed,ax00_sed_cb) 
    ax00.set_title(r'$Mn2\ Farm\ 3x,\ \mu M\ $')   

    plot_param('Mn2',get_z(df_brom_5,'O2'),ax00_1,ax00_1_cb,ax00_1_sed,ax00_1_sed_cb) 
    ax00_1.set_title(r'$Mn2\ Farm\ 5x,\ \mu M\ $')

    plot_param('Fe2',get_z(df_brom_3,'Fe2'),ax01,ax01_cb,ax01_sed,ax01_sed_cb) 
    ax01.set_title(r'$Fe2\ Farm\ 3x,\ \mu M\ $')

    plot_param('Fe2',get_z(df_brom_5,'Fe2'),ax01_1,ax01_1_cb,ax01_1_sed,ax01_1_sed_cb) 
    ax01_1.set_title(r'$Fe2\ Farm\ 5x,\ \mu M\ $')

    plot_param('H2S',get_z(df_brom_3,'H2S'),ax02,ax02_cb,ax02_sed,ax02_sed_cb) 
    ax02.set_title(r'$H_2S\ Farm\ 3x,\ \mu M\ $')

    plot_param('H2S',get_z(df_brom_5,'H2S'),ax02_1,ax02_1_cb,ax02_1_sed,ax02_1_sed_cb) 
    ax02_1.set_title(r'$H_2S\ Farm\ 5x,\ \mu M\ $')

     
    [axis.set_ylabel('Depth,m') for axis in [ax00,ax01,ax02]]
    [axis.set_ylabel('Depth,cm') for axis in [ax00_sed,ax01_sed,ax02_sed]] 

    
    plt.savefig('Results/Figure9.png')
    #plt.show()


if __name__ == '__main__':
    #fig7()    
    
    fig9()

    