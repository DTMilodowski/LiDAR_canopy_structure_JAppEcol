###############################################################################################################
# This driver function analyses both LiDAR data and field inventory data to produce independent estimates of
# canopy structure.  These are compared against each other and their integrated LAD is compared against LAI
# estimates from hemispherical photographs.
###############################################################################################################
import numpy as np
from scipy import stats
import sys
import canopy_structure_plots as csp
import inventory_based_LAD_profiles as field
import LiDAR_io as io

#------------------------------------------------------------------------------------
# DIRECTORIES
# start by defining input files
subplot_coordinate_file = 'BALI_subplot_coordinates_corrected.csv'
allometry_file = '/home/dmilodow/DataStore_DTM/BALI/LiDAR/Data/Regional/Allometry/Crown_depth_data_SEAsia.csv'
field_file = '/home/dmilodow/DataStore_DTM/BALI/LiDAR/Data/Local/SAFE_DANUM_carbonplots_FieldMapcensus2016.csv'
gps_pts_file = 'GPS_points_file_for_least_squares_fitting.csv'

# also define output directory (for saving figures)
data_dir = '/home/dmilodow/DataStore_DTM/BALI/PAPERS/PaperDrafts/EstimatingCanopyStructureBALI/Profiles/'
output_dir = '/home/dmilodow/DataStore_DTM/BALI/PAPERS/PaperDrafts/EstimatingCanopyStructureBALI/FiguresRevised/v2_5/'

#------------------------------------------------------------------------------------
# PARAMETERS
# define important parameters for canopy profile estimation
Plots = ['LF','E','Belian','Seraya','B North','B South','DC1','DC2']
#Plots = ['B North']
N_plots = len(Plots)
n_subplots = 100
max_height = 80
layer_thickness = 1.
n_layers = np.ceil(max_height/layer_thickness)
minimum_height = 2.
plot_area = 10.**4
subplot_area = 10.*10.
kappa_i = 0.70
kappa = 0.50
kappa_scalar = kappa_i/kappa

heights = np.arange(0,max_height,layer_thickness)+layer_thickness
heights_rad = np.arange(0,max_height+layer_thickness,layer_thickness)

#------------------------------------------------------------------------------------
# LOADING DATA
# load field data and retrieve allometric relationships
field_data = field.load_crown_survey_data(field_file)

# Load LiDAR point clouds for the plots
plot_point_cloud= np.load('%s/10m_grid/plot_point_clouds_v2_5.npz' % data_dir,allow_pickle=True)['arr_0'][()]

# Load LiDAR canopy profiles
PAD,lidar_profiles = np.load('%s/10m_grid/lidar_PAD_profiles_adaptive_20m_grid_v2_5.npz' % data_dir,allow_pickle=True)['arr_0'][()]
PAD_mean = {}
for pp in range(0,N_plots):
    PAD_mean[Plots[pp]] = np.nansum(PAD[Plots[pp]],axis=0)/(np.sum(np.isfinite(PAD[Plots[pp]]),axis=0)).astype('float')

# Load LiDAR PAI
PAI = np.load('%s/10m_grid/lidar_PAI_adaptive_20m_grid_v2_5.npz' % data_dir,allow_pickle=True)['arr_0'][()]

# Load Inventory profiles
temp = np.load('%sinventory_canopy_profiles.npz' % data_dir,allow_pickle=True)['arr_0'][()]
inventory_PAD_temp=temp[0]
inventory_PAD_std=temp[1]
inventory_PAI=temp[2]
inventory_PAD_all_temp=temp[3]
temp = None
inventory_PAD = {}
inventory_PAD_all = {}
bPlots = [b'LF',b'E',b'Belian',b'Seraya',b'B North',b'B South',b'DC1',b'DC2']
for ii, plot in enumerate(Plots):
    inventory_PAD[plot]=inventory_PAD_temp[bPlots[ii]].copy()
    inventory_PAD_all[plot]=inventory_PAD_all_temp[bPlots[ii]].copy()
    PAI[plot]=PAI[plot]*kappa_scalar
    PAD_mean[plot]=PAD_mean[plot]*kappa_scalar
    PAD[plot]=PAD[plot]*kappa_scalar

#===============================================================================
# NOW MAKE PLOTS

#-------------------------------
# INTRODUCTION & METHODS
#-------------------------------
"""
# Figure 1 - Location map, with Hansen data and plot locations
"""
figure_name = output_dir+'Fig1_Location_map.png'
figure_number = 1
csp.plot_location_map(figure_name,figure_number)

"""
# Figure 2 sample point cloud - coloured by return number

figure_name = output_dir+'Fig2_sample_point_cloud.png'
figure_number = 2
csp.plot_point_cloud(figure_name,figure_number,gps_pts_file,plot_point_cloud)
"""

#-------------------------------
# RESULTS - STRUCTURAL CHANGES
#           ACROSS GRADIENT
#-------------------------------
"""
# Figure 3 - PAI plotted against basal area
"""
figure_name = output_dir + 'Fig3_PAI_vs_basal_area_20m_grid.png'
figure_number = 3

# Basal area (m^2 / ha) and standard errors
# data from table 1 of Riutta et al, GCB, 2018
BA = {}
BA['Belian']=(41.6,3.59)
BA['Seraya']=(34.7,2.74)
BA['LF']= (19.3,1.7)
BA['E']= (19.6,1.88)
BA['B North']=(11.1,1.81)
BA['B South']=(6.81,1.00)
BA['DC1']=(32.0,3.3)
BA['DC2']=(30.6,3.37)

colour = ['#46E900','#1A2BCE','#E0007F']
plot_colour = {}
plot_colour['Belian']=colour[0];plot_colour['Seraya']=colour[0];plot_colour['DC1']=colour[0]
plot_colour['DC2']=colour[0];plot_colour['LF']=colour[1];plot_colour['E']=colour[1]
plot_colour['B North']=colour[2];plot_colour['B South']=colour[2]

plot_marker = {}
plot_marker['Belian']='o';plot_marker['Seraya']='v';plot_marker['DC1']='^';plot_marker['DC2']='s'
plot_marker['LF']='o';plot_marker['E']='v';plot_marker['B North']='o';plot_marker['B South']='v'
plot_label = {}
plot_label['Belian']='MLA01';plot_label['Seraya']='MLA02';plot_label['DC1']='DAN04';plot_label['DC2']='DAN05'
plot_label['LF']='SAF04';plot_label['E']='SAF03';plot_label['B North']='SAF02';plot_label['B South']='SAF01'
csp.plot_PAI_vs_basal_area_single(figure_name,figure_number,PAD,PAD_mean,BA,
                            plot_marker,plot_label,plot_colour,layer_thickness=1)

"""
# Figure 4 - Point clouds and profiles across degradation gradient
"""
figure_name = output_dir + 'Fig4_pointclouds_and_profiles_20m_grid.png'
figure_number = 4
gps_pts_file = 'GPS_points_file_for_least_squares_fitting.csv'
csp.plot_point_clouds_and_profiles_single(figure_name,figure_number, gps_pts_file,
                        plot_point_cloud,heights,lidar_profiles,
                        PAD,PAD_mean,inventory_PAD,inventory_PAD_all)

"""
# Figure 5 - Sensitivity of PAI estimates to pulse density and resolution
# Figure 6 - Sensitivity analysis of unsampled voxels
# see sensitivity_analysis_figures_revised
"""
figure_name = output_dir + 'Fig5_PAI_sensitivity.png'

"""
# Figure 7 - Niche availability
# Plotting Shannon Index (panel a) and overstory PAD (panel b)
"""
figure_name = output_dir + 'Fig7_niche_availability_20m_grid.png'
figure_number = 7
csp.plot_niche_availability(figure_name,figure_number,PAD,heights)

figure_name = output_dir + 'Fig7_shannon_diversity_20m_grid.png'
csp.plot_shannon_diversity_index_distributions(figure_name,PAD,heights)
figure_name = output_dir + 'Fig8_light_environmetns_20m_grid.png'
csp.plot_subcanopy_environments(figure_name,PAD,heights)

"""
#-------------------------------
# SUPPLEMENT
#-------------------------------
# Figure S1 - example crown model
"""
field_data = field.load_crown_survey_data(field_file)
a, b, CF, r_sq, p, H, D, H_i, PI_u, PI_l = field.retrieve_crown_allometry(allometry_file)
a_ht, b_ht, CF_ht, a_A, b_A, CF_A = field.calculate_allometric_equations_from_survey(field_data)
figure_number = 111
figure_name = output_dir+'figS1_crown_model_example'
Plot_name = b'Belian'
angle = 45.
csp.plot_canopy_model(figure_number,figure_name,Plot_name,field_data,angle,
a_ht, b_ht, CF_ht, a_A, b_A, CF_A, a, b, CF)


"""
# Figure S2 - Allometric models; include confidence intervals, and add vertical band
# illustrating the 10 cm DBH cutoff
"""
figure_name = output_dir + 'FigS2_allometric_relationships.png'
figure_number = 112
csp.plot_allometric_relationships(figure_name,figure_number,field_file,allometry_file)


"""
# Figure S6 comparison of profiles for the two Danum sites
"""
figure_number = 116
figure_name = output_dir+'FigS6_pointclouds_and_profiles_20m_grid_Danum.png'
csp.plot_point_clouds_and_profiles_Danum_single(figure_name,figure_number, gps_pts_file,
                        plot_point_cloud,heights, lidar_profiles,
                        PAD,PAD_mean,inventory_PAD,inventory_PAD_all)

"""
# Figure S7 - sensitivity of 1 ha profiles to pulse density
# Figure S8 - relative CIs across profiles for different pulse densities
# Figure S9 - sensitivity of 1 ha profiles to grid resolution
# Figure S10- relative CIs across profiles for different grid resolutions
# see sensitivity_analysis_figures_revised
"""
"""
#===============================================================================
# Summary statistics
"""
table_plots = ['Belian','Seraya','DC1','DC2','E','LF','B North','B South']
btable_plots = [b'Belian',b'Seraya',b'DC1',b'DC2',b'E',b'LF',b'B North',b'B South']
print("Plot    \tPAI\t+/-\tcv\t+/-")
cv_=[]
pai_=[]
for pp,plot in enumerate(table_plots):
    bplot=btable_plots[pp]
    pai = np.mean(PAI[plot])
    pai_.append(pai)
    pai_s = stats.sem(PAI[plot])
    cv = np.mean(inventory_PAI[bplot])
    cv_.append(cv)
    cv_s = stats.sem(inventory_PAI[bplot])

    print('%s     \t%.1f\t%.2f\t%.1f\t%.2f\t' % (plot,pai,pai_s,cv,cv_s))

# Area under curve analysis
for pp,plot in enumerate(table_plots):
    pad = PAD[plot][:,2:].mean(axis=0)
    inv = inventory_PAD[plot][2:]
    # normalise by total in column
    pad = pad/np.sum(pad)
    inv = inv/np.sum(inv)

    AUC = np.sum(np.min((pad,inv),axis=0))*100
    print('%.02f' % AUC)
