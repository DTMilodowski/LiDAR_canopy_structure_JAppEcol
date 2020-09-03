from matplotlib import pyplot as plt
import numpy as np
from math import pi 

box_labels = {}
box_labels["maliau_belian"] = ["01", "02", "03", "04", "05", "10", "09", "08", "07", "06", "11", "12", "13", "14", "15", "20", "19", "18", "17", "16", "21", "22", "23", "24", "25"]
box_labels["B_north"] = ["01", "02", "03", "04", "05", "10", "09", "08", "07", "06", "11", "12", "13", "14", "15", "20", "19", "18", "17", "16", "21", "22", "23", "24", "25"]
box_labels["B_south"] = ["01", "02", "03", "04", "05", "10", "09", "08", "07", "06", "11", "12", "13", "14", "15", "20", "19", "18", "17", "16", "21", "22", "23", "24", "25"]
box_labels["LFE"] = ["01", "02", "03", "04", "05", "10", "09", "08", "07", "06", "11", "12", "13", "14", "15", "20", "19", "18", "17", "16", "21", "22", "23", "24", "25"]
box_labels["maliau_seraya"] = ["01", "02", "03", "04", "05", "06", "11", "10", "09", "08", "07", "_", "12", "13", "14", "15", "16", "_", "21", "20", "19", "18", "17", "_", "22", "23", "24", "25","_","_"]
box_labels["E"] = ["01","02","03","04","05","06","07","14","13","12","11","10","09","08","_","_","15","16","17","18","19","_","_","22","21","20","_","_","_","_","23","24","25","_","_"]
box_labels["danum_1"] = ["01", "02", "03", "04", "05", "10", "09", "08", "07", "06", "11", "12", "13", "14", "15", "20", "19", "18", "17", "16", "21", "22", "23", "24", "25"]
box_labels["danum_2"] = ["01", "02", "03", "04", "05", "10", "09", "08", "07", "06", "11", "12", "13", "14", "15", "20", "19", "18", "17", "16", "21", "22", "23", "24", "25"]


#C_plots
outfile = "Cplots_subplot_coordinates.csv"
plot = ["maliau_belian", "maliau_seraya", "B_north", "B_south", "LFE"]
bbox_corners_x = [[496613.2, 496689.7, 496761.2, 496688.3],[494459.7, 494458.9, 494558.9, 494558.9],
                  [568428.9, 568428.5, 568534.7, 568530.1],[568537.9, 568538.3, 568638.8, 568638.7],
                  [577581.4, 577581, 577679.8, 577677]]

bbox_corners_y = [[524716.6, 524785., 524715.1, 524643.6],[525307.2, 525409.1, 525409.1, 525307.2],
                  [523821.8, 523920.8, 523920.5, 523826.1],[523005.9, 523107.8, 523107.3, 523004.3],
                  [526681.6, 526780.4, 526784, 526683.4]]

N_deg = [315., 0.,  0., 0., 230.]

"""
# Danum plots
outfile = "Danum_subplot_coordinates.csv"
plot = ["danum_1", "danum_2"]
bbox_corners_x = [[588114.4341, 588115.1391, 588224.5788, 588222.0747], [587955.6, 587954.8, 588055.9, 588056.8]]

bbox_corners_y = [[547490.3246, 547591.0093, 547591.0093, 547490.3386], [547518.2, 547619.9, 547619.9, 547519.3]]

N_deg = [270., 270.]


# E plot
outfile = "E_subplot_coordinates.csv"
plot = ["E"]
bbox_corners_x = [[565171.6]]
bbox_corners_y = [[518647.5]]
N_deg = [0.]
"""
n_plots = len(plot)

subplot_bbox = []
subplot_bbox_alt = []
subplot_labels=[]
 
for pp in range(0,n_plots):
    x_centre = np.mean(bbox_corners_x[pp])
    y_centre = np.mean(bbox_corners_y[pp])

    N_rad = pi*N_deg[pp]/180

    plot_width = 100.
    subplot_width = 20

    rows = 5
    cols = 5
    if plot[pp]=="maliau_seraya":
        cols = 6
    if plot[pp]=='E':
        cols = 7

    x_prime=np.arange(0,cols+1.)*subplot_width-plot_width/2.
    y_prime=np.arange(0,rows+1.)*subplot_width-plot_width/2.

    xv_prime,yv_prime=np.asarray(np.meshgrid(x_prime,y_prime))

    xv_prime=xv_prime.reshape(xv_prime.size)
    yv_prime=yv_prime.reshape(yv_prime.size)

    xy_prime=np.asarray([xv_prime,yv_prime])

    rotation_matrix = np.asarray([[np.cos(-N_rad), -np.sin(-N_rad)],[np.sin(-N_rad), np.cos(-N_rad)]])
    xy=np.dot(rotation_matrix,xy_prime)
    xy_pos = xy+np.asarray([[x_centre],[y_centre]])

    x_grid = xy_pos[0].reshape(rows+1,cols+1)
    y_grid = xy_pos[1].reshape(rows+1,cols+1)

    count = 0
    subplot=[]
    for i in range(0,rows):
        for j in range(0,cols):
            bbox = [ [x_grid[i,j], x_grid[i+1,j], x_grid[i+1,j+1], x_grid[i,j+1], x_grid[i,j]],
                     [y_grid[i,j], y_grid[i+1,j], y_grid[i+1,j+1], y_grid[i,j+1], y_grid[i,j]] ]
            if box_labels[plot[pp]][count]!="_":
                subplot_bbox.append(bbox)
            bbox_alt = [ x_grid[i,j], y_grid[i,j], x_grid[i+1,j], y_grid[i+1,j], x_grid[i+1,j+1],
                         y_grid[i+1,j+1], x_grid[i,j+1], y_grid[i,j+1], x_grid[i,j], y_grid[i,j] ]
            if box_labels[plot[pp]][count]!="_":
                subplot_bbox_alt.append(bbox_alt)
                subplot.append(box_labels[plot[pp]][count])
            count+=1
    bboxes = np.asarray(subplot_bbox)
    plt.figure(pp+1, facecolor='White',figsize=[10,10])
    ax1 = plt.subplot2grid((1,1),(0,0))
    """
    plt.plot(xy_pos[0,:],xy_pos[1,:],'+')
    plt.plot(xy_pos[0,0],xy_pos[1,0],'o')
    """
    for i in range(0,25):#rows*cols):
        #ax1.plot(xy_pos[0,:],xy_pos[1,:],'+')
        #ax1.plot(xy_pos[0,0],xy_pos[1,0],'o')
        ax1.plot(bboxes[25*pp+i,0,:],bboxes[25*pp+i,1,:],'-')
    
    plt.title(plot[pp])
    plt.show()
    
    # Now need to label the bounding boxes and ignore those that aren't required
    """
    box_labels_int = np.arange(1,rows*cols+1)
    box_labels=[]
    for i in range(0,rows*cols):
        box_labels.append("%03d" % (box_labels_int[i]))    
    """
    N_sub = len(subplot)
    for i in range(0,N_sub):
        #subplot_labels.append((plot[pp]+"_"+box_labels[i]))
        subplot_labels.append((plot[pp]+", "+subplot[i]))
    

f = open(outfile,"w") #opens file
f.write("Subplot, x11, y11, x12, y12, x22, y22, x21, y21, x11, y11\n")
for i in range(0,len(subplot_labels)):
    f.write(subplot_labels[i]+", ")
    for j in range(0,9):
        f.write(str(subplot_bbox_alt[i][j]))
        f.write(", ")
    
    f.write(str(subplot_bbox_alt[i][9]))
    f.write("\n")

f.close()

