import numpy as np
import sensitivity_analysis_plots as sp

output_dir = '/home/dmilodow/DataStore_DTM/BALI/PAPERS/PaperDrafts/EstimatingCanopyStructureBALI/FiguresRevised/201910/'

# Load in the sensitivity analysis results
max_height=80
heights = np.arange(0.,max_height)+1
kappa_i = 0.70
kappa = 0.50
kappa_scalar = kappa_i/kappa

resolution_wt_Belian = np.load('sensitivity_analysis_profiles/MH_resolution_wt_sensitivity_adaptive_Belian_sensitivity2.npy',allow_pickle=True)[()]
resolution_wt_BNorth = np.load('sensitivity_analysis_profiles/MH_resolution_wt_sensitivity_adaptive_BNorth_sensitivity2.npy',allow_pickle=True)[()]
resolution_wt_E = np.load('sensitivity_analysis_profiles/MH_resolution_wt_sensitivity_adaptive_E_sensitivity2.npy',allow_pickle=True)[()]
"""
density_wt_Belian_10m = np.load('sensitivity_analysis_profiles/MH_wt_density_sensitivity_Belian_10m_grid.npy',allow_pickle=True)[()]
density_wt_BNorth_10m = np.load('sensitivity_analysis_profiles/MH_wt_density_sensitivity_BNorth_10m_grid.npy',allow_pickle=True)[()]
density_wt_E_10m = np.load('sensitivity_analysis_profiles/MH_wt_density_sensitivity_E_10m_grid.npy',allow_pickle=True)[()]

density_wt_Belian_05m = np.load('sensitivity_analysis_profiles/MH_wt_density_sensitivity_Belian_5m_grid.npy',allow_pickle=True)[()]
density_wt_BNorth_05m = np.load('sensitivity_analysis_profiles/MH_wt_density_sensitivity_BNorth_5m_grid.npy',allow_pickle=True)[()]
density_wt_E_05m = np.load('sensitivity_analysis_profiles/MH_wt_density_sensitivity_E_5m_grid.npy',allow_pickle=True)[()]
"""
density_wt_Belian_20m = np.load('MH_wt_density_sensitivity_Belian_20m_grid.npy',allow_pickle=True)[()]
density_wt_BNorth_20m = np.load('MH_wt_density_sensitivity_BNorth_20m_grid.npy',allow_pickle=True)[()]
density_wt_E_20m = np.load('MH_wt_density_sensitivity_E_20m_grid.npy',allow_pickle=True)[()]

for res in resolution_wt_Belian.keys():
    resolution_wt_Belian[res]['25']*=kappa_scalar
    resolution_wt_E[res]['25']*=kappa_scalar
    resolution_wt_BNorth[res]['25']*=kappa_scalar

for den in density_wt_E_20m['20m'].keys():
    density_wt_Belian_20m['20m'][den]*=kappa_scalar
    density_wt_E_20m['20m'][den]*=kappa_scalar
    density_wt_BNorth_20m['20m'][den]*=kappa_scalar

#-------------------------------
# RESULTS - SENSITIVITY ANALYSIS
#-------------------------------
"""
# Figure 5 - Sensitivity of PAI estimates to pulse density and resolution
"""
"""
figure_name = output_dir + 'Fig5_PAI_sensitivity.png'
figure_number = 5
sp.plot_PAI_sensitivity(figure_number,figure_name,
                        resolution_MH_Belian,resolution_MH_BNorth,resolution_MH_E,
                        resolution_rad_Belian,resolution_rad_BNorth,resolution_rad_E,
                        resolution_wt_Belian,resolution_wt_BNorth,resolution_wt_E,
                        density_MH_Belian,density_MH_BNorth,density_MH_E,
                        density_rad_Belian,density_rad_BNorth,density_rad_E,
                        density_wt_Belian,density_wt_BNorth,density_wt_E)
"""
figure_name = output_dir + "fig5_PAI_sensitivity_to_pulse_density_m1b_20m.png"
sp.plot_PAI_sensitivity_density_single(figure_name,density_wt_Belian_20m,
                            density_wt_BNorth_20m,density_wt_E_20m)
"""
figure_name = output_dir + "fig5_PAI_sensitivity_to_pulse_density_m1b_10m.png"
sp.plot_PAI_sensitivity_density_single(figu_number,figure_name,density_wt_Belian_10m,
                            density_wt_BNorth_10m,density_wt_E_10m)
figure_name = output_dir + "fig5_PAI_sensitivity_to_pulse_density_m1b_05m.png"
sp.plot_PAI_sensitivity_density_single(figure_name,density_wt_Belian_05m,
                            density_wt_BNorth_05m,density_wt_E_05m)
"""

figure_name = output_dir + "fig6_PAI_sensitivity_to_resolution.png"
sp.plot_PAI_sensitivity_resolution_single(figure_name,resolution_wt_Belian,
                            resolution_wt_BNorth,resolution_wt_E)

"""
#-------------------------------
# SUPPLEMENT
#-------------------------------
# Figure S7 - Sensitivity analysis of vertical profiles to point density
"""
figure_number = 117
figure_name = output_dir + "figS7_profile_sensitivity_to_pulse_density_m1b_20m.png"
sp.plot_profile_sensitivity_density_single(figure_number,figure_name,heights,density_wt_Belian_20m,
                            density_wt_BNorth_20m,density_wt_E_20m,res_key='20m')
"""
figure_name = output_dir + "figS7_profile_sensitivity_to_pulse_density_m1b_10m.png"
sp.plot_profile_sensitivity_density_single(figure_number,figure_name,heights,density_wt_Belian_10m,
                            density_wt_BNorth_10m,density_wt_E_10m,res_key='10m')
figure_name = output_dir + "figS7_profile_sensitivity_to_pulse_density_m1b_05m.png"
sp.plot_profile_sensitivity_density_single(figure_number,figure_name,heights,density_wt_Belian_05m,
                            density_wt_BNorth_05m,density_wt_E_05m,res_key='05m')
"""

"""
# Figure S8 - sensitivity analysis, confidence interval sensitivity to density
"""
figure_number = 118
figure_name = output_dir + "figS8_profile_sensitivity_to_point_density_individual_CI_10m.png"
sp.plot_profile_sensitivity_to_point_density_individual_CI_single(figure_name,heights,density_wt_Belian_20m)

"""
figure_name = output_dir + "figS8_profile_sensitivity_to_point_density_individual_CI.png"
sp.plot_profile_sensitivity_to_point_density_individual_CI(figure_number,figure_name,heights,
                    density_MH_Belian,density_rad_Belian,density_wt_Belian)
figure_name = output_dir + "figS8_profile_sensitivity_to_point_density_individual_CI_10m.png"
sp.plot_profile_sensitivity_to_point_density_individual_CI_single(figure_name,heights,density_wt_Belian_10m)
"""
"""
 Figure S9 - Sensitivity analysis of vertical profiles to spatial resolution
 Comparison of OG vs Moderately Logged vs. Heavily Logged
"""
figure_number = 119
figure_name = output_dir + "figS9_profile_sensitivity_to_resolution_adaptive_sensitivity_m1b.png"
sp.plot_profile_sensitivity_resolution_single(figure_number,figure_name,heights,resolution_wt_Belian,
                            resolution_wt_BNorth,resolution_wt_E)
"""
# Figure S10 - sensitivity analysis, confidence interval sensitivity to resolution
"""
figure_number = 1110
figure_name = output_dir + "figS10_profile_sensitivity_to_resolution_individual_CI.png"
sp.plot_profile_sensitivity_to_resolution_individual_CI(figure_number,figure_name,heights,
                    resolution_MH_Belian,resolution_rad_Belian,resolution_wt_Belian)
figure_name = output_dir + "figS10_profile_sensitivity_to_resolution_individual_CI_single.png"
sp.plot_profile_sensitivity_to_resolution_individual_CI_single(figure_name,heights,resolution_wt_Belian)




"""
resolution_MH_Belian = np.load('sensitivity_analysis_profiles/MH_resolution_sensitivity_adaptive_Belian.npy')[()]
resolution_MH_BNorth = np.load('sensitivity_analysis_profiles/MH_resolution_sensitivity_adaptive_BNorth.npy')[()]
resolution_MH_E = np.load('sensitivity_analysis_profiles/MH_resolution_sensitivity_adaptive_E.npy')[()]

resolution_rad_Belian = np.load('sensitivity_analysis_profiles/rad2_resolution_sensitivity_adaptive_Belian_sensitivity2.npy')[()]
resolution_rad_BNorth = np.load('sensitivity_analysis_profiles/rad2_resolution_sensitivity_adaptive_BNorth_sensitivity2.npy')[()]
resolution_rad_E = np.load('sensitivity_analysis_profiles/rad2_resolution_sensitivity_adaptive_E_sensitivity2.npy')[()]

penetration_lim_Belian = np.load('sensitivity_analysis_profiles/penetration_limit_resolution_adaptive_Belian.npy')[()]
penetration_lim_BNorth = np.load('sensitivity_analysis_profiles/penetration_limit_resolution_adaptive_BNorth.npy')[()]
penetration_lim_E = np.load('sensitivity_analysis_profiles/penetration_limit_resolution_adaptive_E.npy')[()]

density_MH_Belian = np.load('sensitivity_analysis_profiles/MH_density_sensitivity_Belian.npy')[()]
density_MH_BNorth = np.load('sensitivity_analysis_profiles/MH_density_sensitivity_BNorth.npy')[()]
density_MH_E = np.load('sensitivity_analysis_profiles/MH_density_sensitivity_E.npy')[()]

density_rad_Belian = np.load('sensitivity_analysis_profiles/rad2_density_sensitivity_Belian_sensitivity2.npy')[()]
density_rad_BNorth = np.load('sensitivity_analysis_profiles/rad2_density_sensitivity_BNorth_sensitivity2.npy')[()]
density_rad_E = np.load('sensitivity_analysis_profiles/rad2_density_sensitivity_E_sensitivity2.npy')[()]

density_wt_Belian = np.load('sensitivity_analysis_profiles/MH_wt_density_sensitivity_Belian.npy')[()]
density_wt_BNorth = np.load('sensitivity_analysis_profiles/MH_wt_density_sensitivity_BNorth.npy')[()]
density_wt_E = np.load('sensitivity_analysis_profiles/MH_wt_density_sensitivity_E.npy')[()]

figure_name = output_dir + "figS7_profile_sensitivity_to_pulse_density.png"
sp.plot_profile_sensitivity_density(figure_number,figure_name,heights,density_MH_Belian,
                            density_MH_BNorth,density_MH_E,density_rad_Belian,
                            density_rad_BNorth,density_rad_E,density_wt_Belian,
                            density_wt_BNorth,density_wt_E)


figure_name = output_dir + "figS9_profile_sensitivity_to_resolution_adaptive_sensitivity.png"
sp.plot_profile_sensitivity_resolution_full(figure_number,figure_name,heights,resolution_MH_Belian,
                            resolution_MH_BNorth,resolution_MH_E,resolution_rad_Belian,
                            resolution_rad_BNorth,resolution_rad_E,resolution_wt_Belian,
                            resolution_wt_BNorth,resolution_wt_E)
"""
