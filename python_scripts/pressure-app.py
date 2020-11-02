import pandas as pd
from pandas import DataFrame as df
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wntr as wn
import networkx as nx
from collections import defaultdict
import tensorflow as tf
import math

in2_ft2 = 0.0005787

def pressDeviation(file0, file1):
    res_arr = file1
    res_arr['PressureDeviation']= file1.NodePressure
    res_arr['FlowDeviation']= file1.NodeResultFlow
    res_arr.FlowDeviation = abs(res_arr.NodeResultFlow.subtract(file0.NodeResultFlow))/file0.NodeResultFlow
    res_arr.PressureDeviation = abs(res_arr.PressureDeviation.subtract(file0.NodePressure))/file0.NodePressure
    return res_arr

def reducer(input_df, template_df):
    unique_nodes = template_df.NAME.unique()
    reduced_nodeArr = input_df[input_df.NAME.isin(unique_nodes)]
    reduced_nodeArr.reset_index(inplace = True, drop = True)
    return reduced_nodeArr

color_picker=np.array([[0,0,0,0,0],[0,1,1,2,2],[1,2,2,3,3],[2,3,3,4,4],[3,4,4,4,4],[4,4,4,4,4]])
def color_indexer(input_arr):
    out_arr=np.array(input_arr)
    for i in range(0,input_arr[0].size):
        out_arr[0,i]=color_picker[input_arr[0,i]][0]
        out_arr[1,i]=color_picker[input_arr[1,i]][1]
        out_arr[2,i]=color_picker[input_arr[2,i]][2]
        out_arr[3,i]=color_picker[input_arr[3,i]][3]
        out_arr[4,i]=color_picker[input_arr[4,i]][4]

    return out_arr

scaled_vals=np.array([res0_1_200.PressureDeviation ,res0_11_200.PressureDeviation,res0_21_200.PressureDeviation,res0_31_200.PressureDeviation, res0_41_200.PressureDeviation])
scaled_vals[scaled_vals > 0.24] = 5
scaled_vals[scaled_vals <=0.03] = 6
scaled_vals[scaled_vals <=0.06] = 1
scaled_vals[scaled_vals <=0.12] = 2
scaled_vals[scaled_vals <=0.18] = 3
scaled_vals[scaled_vals <=0.24] = 4
scaled_vals[scaled_vals == 6] = 0
scaled_vals=scaled_vals.astype(int)
colors_array = color_indexer(scaled_vals)
color_indexed0_41_200=res0_41_200
color_indexed0_41_200['color']=colors_array[4]
color_indexed0_31_200=res0_31_200
color_indexed0_31_200['color']=colors_array[3]
color_indexed0_21_200=res0_21_200
color_indexed0_21_200['color']=colors_array[2]
color_indexed0_11_200=res0_11_200
color_indexed0_11_200['color']=colors_array[1]
color_indexed0_1_200=res0_1_200
color_indexed0_1_200['color']=colors_array[0]

def save_data_file_200(temp):
    if temp == 16:
        arr1=color_indexed1_200
        arr11=color_indexed11_200
        arr21=color_indexed21_200
        arr31=color_indexed31_200
        arr41=color_indexed41_200
    elif temp==0:
        arr1=color_indexed0_1_200
        arr11=color_indexed0_11_200
        arr21=color_indexed0_21_200
        arr31=color_indexed0_31_200
        arr41=color_indexed0_41_200
    elif temp==32:
        arr1=color_indexed32_1_200
        arr11=color_indexed32_11_200
        arr21=color_indexed32_21_200
        arr31=color_indexed32_31_200
        arr41=color_indexed32_41_200
    elif temp==48:
        arr1=color_indexed48_1_200
        arr11=color_indexed48_11_200
        arr21=color_indexed48_21_200
        arr31=color_indexed48_31_200
        arr41=color_indexed48_41_200
    elif temp==64:
        arr1=color_indexed64_1_200
        arr11=color_indexed64_11_200
        arr21=color_indexed64_21_200
        arr31=color_indexed64_31_200
        arr41=color_indexed64_41_200

    day = np.hstack([[0]*arr1.NAME.size,[1]*arr1.NAME.size,[2]*arr1.NAME.size,[3]*arr1.NAME.size,[4]*arr1.NAME.size])
    p = np.hstack([arr1.NodePressure,arr11.NodePressure,arr21.NodePressure,arr31.NodePressure,arr41.NodePressure])
    pd = np.hstack([arr1.PressureDeviation,arr11.PressureDeviation,arr21.PressureDeviation,arr31.PressureDeviation,arr41.PressureDeviation])
    names = np.hstack([res1_200.NAME,res11_200.NAME,res21_200.NAME,res31_200.NAME,res41_200.NAME])
    temps=np.hstack([[temp]*arr1.NAME.size,[temp]*arr1.NAME.size,[temp]*arr1.NAME.size,[temp]*arr1.NAME.size,[temp]*arr1.NAME.size])
    color=np.hstack([arr1.color,arr11.color,arr21.color,arr31.color,arr41.color])
    final_file = np.vstack([names.T,p.T,pd.T,temps.T,color.T,day.T])
    ff = np.vstack([p,pd,day])
    #final_file.to_excel("./data/data_nodes_press_200_"+str(temp)+".xlsx")
    #np.savetxt("data_pipes_vel_200"+str(temp)+".txt", final_file.T,fmt='%s')
    #np.savetxt("./data/nodes/data_nodes_press_200_"+str(temp)+".csv",ff,delimiter=",",fmt='%s')


    labels = np.hstack([[0]*arr1.NAME.size,[0]*arr1.NAME.size,[0]*arr1.NAME.size,[0]*arr1.NAME.size,[0]*arr1.NAME.size])
    labels[np.where(final_file[4]==4)]=1
    targets=labels

    #np.savetxt("target_nodes_200"+str(temp)+".txt", targets.T,fmt='%s')
    np.savetxt("/Users/aya/Documents/code-pfs/gas-nx/export/dP_200nodes"+str(temp)+".csv",targets,delimiter=",",fmt='%s')

    return final_file, targets
