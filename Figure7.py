import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
'''
Scrip to plot Figure 7,8
Distance Profle
For Farm 1x ,Farm 2x ...
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

h = 0.06
w = 0.05
sed2 = 12

def make_fig_gs(nrows,ncols):
    global fig
    fig = plt.figure(figsize=(8.27,11*nrows/5), dpi=100) 

    gs = gridspec.GridSpec(nrows, ncols) 
    gs.update(left = 0.07,right = 0.93, 
            bottom = 0.08, top = 0.95,
            wspace = 0.25, hspace= 0.42)
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
    try:
        z2 = get_z(df_brom_2,param)
        z3 = get_z(df_brom_3,param)
        sed_levels = util.get_levels(z2[:,sed2:],z3[:,sed2:]) 
        levels = util.get_levels(z2[:,:sed2],z3[:,:sed2])
    except: 
        z = get_z(df_brom,param)
        sed_levels = util.get_levels_1_arr(z[:,sed2:]) 
        levels = util.get_levels_1_arr(z[:,:sed2])
    return levels,sed_levels

def plot_param(param,z,axis,axis_cb,axis_sed,axis_cb_sed):

    levels, sed_levels = get_levels_fig8(param)

    X,Y = np.meshgrid(x,y[:sed2])  
    X_sed,Y_sed = np.meshgrid(x,y_sed[sed2:]) 
    cmap = plt.get_cmap('gist_earth') #cubehelix') #sns.cubehelix_palette(n_colors = 1,as_cmap=True) #'jet') #
    # print (param,vmin,vmax)
    CS_1 = axis.contourf(X,Y, z[:,:sed2].T, levels = levels,extend="both",cmap = cmap) 
    CS_1_sed = axis_sed.contourf(X_sed,Y_sed, z[:,sed2:].T, levels = sed_levels,extend="both",cmap = cmap) 

    tick_locator = ticker.MaxNLocator(nbins=4)

    cb = plt.colorbar(CS_1,cax = axis_cb)

    cb.locator = tick_locator
    cb.update_ticks()

    cb_sed = plt.colorbar(CS_1_sed,cax = axis_cb_sed)

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
def fig7():
    global gs,x,y,y_sed,df_brom

    gs = make_fig_gs(3,2)
    df_brom = get_df(util.path_brom1)

    gs00,gs00_1,gs01,gs01_1,gs02,gs02_1 = call_create_gs([0,1,2,3,4,5])
 
    ax00,ax00_cb, ax00_sed,ax00_sed_cb = sbplt_cb(gs00)  
    ax00_1,ax00_1_cb, ax00_1_sed,ax00_1_sed_cb = sbplt_cb(gs00_1)

    ax01,ax01_cb, ax01_sed,ax01_sed_cb = sbplt_cb(gs01)  
    ax01_1,ax01_1_cb, ax01_1_sed,ax01_1_sed_cb= sbplt_cb(gs01_1)

    ax02,ax02_cb, ax02_sed,ax02_sed_cb = sbplt_cb(gs02)  
    ax02_1,ax02_1_cb, ax02_1_sed,ax02_1_sed_cb= sbplt_cb(gs02_1)

    x = df_brom.i.values
    x = x - x[19] # normalize dist by middle column
    y = df_brom.z.values
    sed = 15 
    y_sed = ((y - y[sed])*100)   


    plot_param('POMR',get_z(df_brom,'POMR'),ax00,ax00_cb,ax00_sed,ax00_sed_cb) 
    ax00.set_title(r'$POMR\ Farm\ 1x,\ \mu M\ N$')   

    plot_param('O2',get_z(df_brom,'O2'),ax00_1,ax00_1_cb,ax00_1_sed,ax00_1_sed_cb) 
    ax00_1.set_title(r'$O_2\ Farm\ 1x, \ \mu M\ N$')

    plot_param('DOMR',get_z(df_brom,'DOMR'),ax01,ax01_cb,ax01_sed,ax01_sed_cb) 
    ax01.set_title(r'$DOMR\ Farm\ 1x,\ \mu M\ $')

    plot_param('NO3',get_z(df_brom,'NO3'),ax01_1,ax01_1_cb,ax01_1_sed,ax01_1_sed_cb) 
    ax01_1.set_title(r'$NO_3\ Farm\ 1x,\ \mu M\ $')

    plot_param('NH4',get_z(df_brom,'NH4'),ax02,ax02_cb,ax02_sed,ax02_sed_cb) 
    ax02.set_title(r'$NH_4\ Farm\ 1x,\ \mu M\ $')

    plot_param('PO4',get_z(df_brom,'PO4'),ax02_1,ax02_1_cb,ax02_1_sed,ax02_1_sed_cb) 
    ax02_1.set_title(r'$PO_4\ Farm\ 1x,\ \mu M\ $')

     
    [axis.set_ylabel('Depth,m') for axis in [ax00,ax01,ax02]]
    [axis.set_ylabel('Depth,cm') for axis in [ax00_sed,ax01_sed,ax02_sed]] 

    plt.savefig('Results/Figure7.png')
    #plt.show()


def fig8():
    global gs,x,y,y_sed

    gs = make_fig_gs(5,2)
    global df_brom_2,df_brom_3
    df_brom_2 = get_df(util.path_brom2)
    df_brom_3 = get_df(util.path_brom3)

    gs00,gs00_1,gs01,gs01_1,gs02,gs02_1,gs03,gs03_1,gs04,gs04_1 = call_create_gs([0,1,2,3,4,5,6,7,8,9])
 
    ax00,ax00_cb, ax00_sed,ax00_sed_cb = sbplt_cb(gs00)  
    ax00_1,ax00_1_cb, ax00_1_sed,ax00_1_sed_cb = sbplt_cb(gs00_1)

    ax01,ax01_cb, ax01_sed,ax01_sed_cb = sbplt_cb(gs01)  
    ax01_1,ax01_1_cb, ax01_1_sed,ax01_1_sed_cb= sbplt_cb(gs01_1)

    ax02,ax02_cb, ax02_sed,ax02_sed_cb = sbplt_cb(gs02)  
    ax02_1,ax02_1_cb, ax02_1_sed,ax02_1_sed_cb= sbplt_cb(gs02_1)

    ax03,ax03_cb, ax03_sed,ax03_sed_cb = sbplt_cb(gs03)  
    ax03_1,ax03_1_cb, ax03_1_sed,ax03_1_sed_cb= sbplt_cb(gs03_1)

    ax04,ax04_cb, ax04_sed,ax04_sed_cb = sbplt_cb(gs04)  
    ax04_1,ax04_1_cb, ax04_1_sed,ax04_1_sed_cb= sbplt_cb(gs04_1)

    x = df_brom_2.i.values
    x = x - x[19] # normalize dist by middle column
    y = df_brom_2.z.values
    sed = 15 
    y_sed = ((y - y[sed])*100)   


    plot_param('POMR',get_z(df_brom_2,'POMR'),ax00,ax00_cb,ax00_sed,ax00_sed_cb) 
    ax00.set_title(r'$POMR\ Farm\ 2x,\ \mu M\ N$')   

    plot_param('POMR',get_z(df_brom_3,'POMR'),ax00_1,ax00_1_cb,ax00_1_sed,ax00_1_sed_cb) 
    ax00_1.set_title(r'$POMR\ Farm\ 3x,\ \mu M\ N$')   


    plot_param('O2',get_z(df_brom_2,'O2'),ax01,ax01_cb,ax01_sed,ax01_sed_cb) 
    ax01.set_title(r'$O_2\ Farm\ 1x, \ \mu M\ N$')

    plot_param('O2',get_z(df_brom_3,'O2'),ax01_1,ax01_1_cb,ax01_1_sed,ax01_1_sed_cb)
    ax01_1.set_title(r'$O_2\ Farm\ 3x, \ \mu M\ N$')

    plot_param('NO3',get_z(df_brom_2,'NO3'),ax02,ax02_cb,ax02_sed,ax02_sed_cb) 
    ax02.set_title(r'$NO_3\ Farm\ 2x,\ \mu M\ $')
    plot_param('NO3',get_z(df_brom_3,'NO3'),ax02_1,ax02_1_cb,ax02_1_sed,ax02_1_sed_cb) 
    ax02_1.set_title(r'$NO_3\ Farm\ 3x,\ \mu M\ $')


    plot_param('NH4',get_z(df_brom_2,'NH4'),ax03,ax03_cb,ax03_sed,ax03_sed_cb) 
    ax03.set_title(r'$NH_4\ Farm\ 2x,\ \mu M\ $')
    plot_param('NH4',get_z(df_brom_3,'NH4'),ax03_1,ax03_1_cb,ax03_1_sed,ax03_1_sed_cb) 
    ax03_1.set_title(r'$NH_4\ Farm\ 3x,\ \mu M\ $')


    plot_param('PO4',get_z(df_brom_2,'PO4'),ax04,ax04_cb,ax04_sed,ax04_sed_cb) 
    ax04.set_title(r'$PO_4\ Farm\ 2x,\ \mu M\ $')
    plot_param('PO4',get_z(df_brom_3,'PO4'),ax04_1,ax04_1_cb,ax04_1_sed,ax04_1_sed_cb) 
    ax04_1.set_title(r'$PO_4\ Farm\ 3x,\ \mu M\ $')


    [axis.set_ylabel('Depth,m') for axis in [ax00,ax01,ax02,ax03,ax04]]
    [axis.set_ylabel('Depth,cm') for axis in [ax00_sed,ax01_sed,ax02_sed]] 

    plt.savefig('Results/Figure8.png')
    #plt.show()

if __name__ == '__main__':
    fig7()    
    
    #fig8()