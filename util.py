import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np


path_brom1 = r'E:\Users\EYA\Hardnew\data_Hard\BROM_Hardangerfjord_1X.nc'
path_brom2 = r'E:\Users\EYA\Hardnew\data_Hard\BROM_Hardangerfjord_out_2X.nc'    
path_brom3 = r'E:\Users\EYA\Hardnew\data_Hard\BROM_Hardangerfjord_out_3X.nc'
path_brom5 = r'E:\Users\EYA\Hardnew\data_Hard\BROM_Hardangerfjord_out_5X.nc'

start_day = 731
stop_day = 1096
dist_day = 831
baseline_col = 0
farm_col = 20
cmap = plt.get_cmap('jet')
nlev = 30

def get_levels_1_arr(arr1):
    vmin = np.min(arr1)
    vmax =  np.max(arr1)
    return np.linspace(vmin,vmax,nlev)


def get_levels(arr1,arr2):
    #nlev = utl200
    vmin = np.min((arr1,arr2))
    vmax =  np.max((arr1,arr2))
    return np.linspace(vmin,vmax,nlev)