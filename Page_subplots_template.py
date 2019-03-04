import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(8.27,11), dpi=100) 

gs = gridspec.GridSpec(5, 2,width_ratios = [30,1]) 
gs.update(left = 0.07,right = 0.93, 
          bottom = 0.04, top = 0.95,
          wspace = 0.05, hspace= 0.3)


h = 0.06
w = 0.15

gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0],hspace=h,wspace=w,height_ratios=[3,2]) 
gs00_cb = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=[3,2]) 

gs01 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[2],hspace=h,wspace=w,height_ratios=[3,2]) 
gs01_cb = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[3],height_ratios=[3,2])

gs02 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[4],hspace=h,wspace=w,height_ratios=[3,2]) 
gs02_cb = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[5],height_ratios=[3,2])

gs03 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[6],hspace=h,wspace=w,height_ratios=[3,2]) 
gs03_cb = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[7],height_ratios=[3,2])

gs04 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[8],hspace=h,wspace=w,height_ratios=[3,2]) 
gs04_cb = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[9],height_ratios=[3,2])

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


plt.show()