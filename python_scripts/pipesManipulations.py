

#######################
## Importing Modules ##
#######################

import pandas as pd
from pandas import DataFrame as df
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wntr as wn
import networkx as nx
from collections import defaultdict
import tensorflow as tf

#######################################
## Populating Anomaly Free DataFrame ##
#######################################

anomalyFree = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Pipes.csv"
nFile0=pd.read_csv(anomalyFree)

anomaly = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Pipes_Leak1.csv"
nFile1=pd.read_csv(anomaly)

#################################
## Creating Anomaly DataFrames ##
#################################

def get_file(name):
    anomaly = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData"+name
    nFile=pd.read_csv(anomaly)
    return nFile

def flowDeviation(file0, file1):
    res_arr = file1
    res_arr['FlowDeviation']= file1.FacilityFlowAbsolute
    res_arr.FlowDeviation = abs(res_arr.FacilityFlowAbsolute.subtract(file0.FacilityFlowAbsolute))/file0.FacilityFlowAbsolute
    return res_arr

def flowDeviation5(file0, file1):
    res_arr = file1
    res_arr['FlowDeviation']= file1.FacilityFlowAbsolute
    res_arr.FlowDeviation = abs(res_arr.FacilityFlowAbsolute.subtract(file0.FacilityFlowAbsolute))/5
    return res_arr

preDir = "/LeakData_ZeroDegrees/"
leaklist = ["NYU Anamoly Data_ZeroDeg_Pipes_Leak11.csv","NYU Anamoly Data_ZeroDeg_Pipes_Leak21.csv", "NYU Anamoly Data_ZeroDeg_Pipes_Leak31.csv", "NYU Anamoly Data_ZeroDeg_Pipes_Leak41.csv"]

leak0_11 = get_file(preDir+leaklist[0])
leak0_21 = get_file(preDir+leaklist[1])
leak0_31 = get_file(preDir+leaklist[2])
leak0_41 = get_file(preDir+leaklist[3])

###########################################
## Calculating Normalized Flow Deviation ##
###########################################

res0_1 = flowDeviation(nFile0,nFile1)
res0_11 = flowDeviation(nFile0,leak0_11)
res0_21 = flowDeviation(nFile0,leak0_21)
res0_31 = flowDeviation(nFile0,leak0_31)
res0_41 = flowDeviation(nFile0,leak0_41)

res0_1 = res0_1.fillna(value=0.0)
res0_11 = res0_11.fillna(value=0.0)
res0_21 = res0_21.fillna(value=0.0)
res0_31 = res0_31.fillna(value=0.0)
res0_41 = res0_41.fillna(value=0.0)

################################################
## Calculating 5(?) Normalized Flow Deviation ##
################################################

res50_1 = flowDeviation5(nFile0,nFile1)
res50_11 = flowDeviation5(nFile0,leak0_11)
res50_21 = flowDeviation5(nFile0,leak0_21)
res50_31 = flowDeviation5(nFile0,leak0_31)
res50_41 = flowDeviation5(nFile0,leak0_41)

res50_1 = res50_1.fillna(value=0.0)
res50_11 = res50_11.fillna(value=0.0)
res50_21 = res50_21.fillna(value=0.0)
res50_31 = res50_31.fillna(value=0.0)
res50_41 = res50_41.fillna(value=0.0)

#################################################
### Finding the Maximum of the Flow Deviation ###
#################################################

max(res0_1.FlowDeviation)

max(res0_11.FlowDeviation)

max(res0_21.FlowDeviation)

max(res0_31.FlowDeviation)

max(res0_41.FlowDeviation)

#res0_1.to_csv(r'/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/pipes1.csv')
#res0_11.to_csv(r'/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/pipes11.csv')
#res0_21.to_csv(r'/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/pipes21.csv')
#res0_31.to_csv(r'/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/pipes31.csv')
#res0_41.to_csv(r'/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/pipes41.csv')

l = abs(res0_41.FacilityFlowAbsolute - nFile0.FacilityFlowAbsolute)
max(l/5)

#from sklearn import svm
#svc = svm.SVC(probability=False,  kernel="linear", C=2.8, gamma=.0073,verbose=10)

###################################################
### Creating the Table of Leak Deviations Scale ###
###################################################

res0_1["leak"] = 0
for row in res0_1.itertuples():
    if row.FlowDeviation > 40:
        row["leak"] = 1

res0_1['leak'] = np.where(res0_1['FlowDeviation']>=40, 1, 0)
res0_11['leak'] = np.where(res0_11['FlowDeviation']>=40, 1, 0)
res0_21['leak'] = np.where(res0_21['FlowDeviation']>=40, 1, 0)
res0_31['leak'] = np.where(res0_31['FlowDeviation']>=40, 1, 0)
res0_41['leak'] = np.where(res0_41['FlowDeviation']>=40, 1, 0)

leakTable=pd.DataFrame(columns=['leak1','leak2','leak3','leak4','leak5'])
leakTable['leak1']=res0_1['leak']
leakTable['leak2']=res0_11['leak']
leakTable['leak3']=res0_21['leak']
leakTable['leak4']=res0_31['leak']
leakTable['leak5']=res0_41['leak']
leakTable['sumLeaks']=leakTable.leak1+leakTable.leak2+leakTable.leak3+leakTable.leak4+leakTable.leak5

arr=range(res0_1.FlowDeviation.size)
plt.plot(arr,leakTable.sumLeaks)

leakTable.sumLeaks.value_counts()

def hightlightColor(r):
    if r['sumLeaks']>3:
        return ['background-color: red']*6
    elif r['sumLeaks']>2:
        return ['background-color: orange']*6
    elif r['sumLeaks']>1:
        return ['background-color: yellow']*6
    elif r['sumLeaks']>0:
        return ['background-color: green']*6
    else:
        return ['background-color: blue']*6

leakTable.style.apply(hightlightColor, axis=1)

###################################################
### Virtual Graph Structure for Pipe Midpoints  ###
###################################################

anomalyFreeNode = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Nodes.csv"
nodeArr=pd.read_csv(anomalyFreeNode)
setOfNames = set(nodeArr['NAME'])

## Leak 41 ##
#create an array of the node information with an appended Zeros matrix#
final_res0_41 = np.array([res0_41['NAME'],res0_41['FacilityFromNodeName'],res0_41['FacilityToNodeName'],res0_41['FlowDeviation'],[0]*res0_41.NAME.size,[0]*res0_41.NAME.size,[0]*res0_41.NAME.size,[0]*res0_41.NAME.size,[0]*res0_41.NAME.size,[0]*res0_41.NAME.size])

for i in range(0,final_res0_41[1].size):
    if final_res0_41[1][i] in setOfNames:
        temp = nodeArr.loc[nodeArr['NAME']==final_res0_41[1][i]]
        final_res0_41[4][i]=temp.iloc[0][3]
        final_res0_41[5][i]=temp.iloc[0][2]
    if final_res0_41[2][i] in setOfNames:
        temp = nodeArr.loc[nodeArr['NAME']==final_res0_41[2][i]]
        final_res0_41[6][i]=temp.iloc[0][3]
        final_res0_41[7][i]=temp.iloc[0][2]

## Leak41 Flow Deviation 5(?) ##
final_res50_41 = np.array([res50_41['NAME'],res50_41['FacilityFromNodeName'],res50_41['FacilityToNodeName'],res50_41['FlowDeviation'],[0]*res50_41.NAME.size,[0]*res50_41.NAME.size,[0]*res50_41.NAME.size,[0]*res50_41.NAME.size,[0]*res50_41.NAME.size,[0]*res50_41.NAME.size])

for i in range(0,final_res50_41[1].size):
    if final_res50_41[1][i] in setOfNames:
        temp = nodeArr.loc[nodeArr['NAME']==final_res50_41[1][i]]
        final_res50_41[4][i]=temp.iloc[0][3]
        final_res50_41[5][i]=temp.iloc[0][2]
    if final_res50_41[2][i] in setOfNames:
        temp = nodeArr.loc[nodeArr['NAME']==final_res50_41[2][i]]
        final_res50_41[6][i]=temp.iloc[0][3]
        final_res50_41[7][i]=temp.iloc[0][2]

#Creating the Virtual Pipe Center point node#
mid_elem_x=(final_res0_41[4]+final_res0_41[6])/2.0
mid_elem_y=(final_res0_41[5]+final_res0_41[7])/2.0
final_res0_41[8]=mid_elem_x
final_res0_41[9]=mid_elem_y
res0_41['mid_point_x']=mid_elem_x
res0_41['mid_point_y']=mid_elem_y

mid_elem5_x=(final_res50_41[4]+final_res50_41[6])/2.0
mid_elem5_y=(final_res50_41[5]+final_res50_41[7])/2.0
final_res50_41[8]=mid_elem5_x
final_res50_41[9]=mid_elem5_y
res50_41['mid_point_x']=mid_elem5_x
res50_41['mid_point_y']=mid_elem5_y

#Plotting 3D Functions#

from mpl_toolkits.mplot3d import Axes3D

def draw_3d(graphArr):
    cntrlnd = '0BEC50B8'
    unique_node=graphArr.FacilityToNodeName.unique()
    u_n_2 = graphArr.FacilityFromNodeName.unique()
    unique_node=np.append(unique_node,u_n_2)
    unique_node=np.unique(unique_node)
    anomalyFreeNode = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Nodes.csv"
    nodeArr=pd.read_csv(anomalyFreeNode)
    nodeArr = nodeArr[nodeArr.NAME.isin(unique_node)]

    G = nx.Graph()

    graphArr['mid_point_names']="mid_point"+graphArr.NAME
    temp_arr = np.array([graphArr.NAME, graphArr.FacilityFromNodeName, graphArr.FacilityToNodeName, graphArr.mid_point_names, graphArr.mid_point_x, graphArr.mid_point_y, [0]*graphArr.NAME.size,[0]*graphArr.NAME.size,[0]*graphArr.NAME.size])
    edges=np.array([['','']])
    m=0
    for i,j in enumerate(temp_arr[3]):
        tmp_src=temp_arr[1][i]
        tmp_dest=temp_arr[2][i]
        srcs = np.where(temp_arr[1]==tmp_src)
        dests = np.where(temp_arr[2] == tmp_src)
        for r in srcs[0]:
            src_node=temp_arr[3][i]
            dest_node=temp_arr[3][r]
            edges = np.vstack([edges, [src_node,dest_node]])
        for k in dests[0]:
            src_node=temp_arr[3][i]
            dest_node=temp_arr[3][k]
            edges = np.vstack([edges, [src_node,dest_node]])
    edges=np.delete(edges,(0), axis=0)

    pos_dict = defaultdict(list)
    mid_pos_dict = defaultdict(list)
    for i, j, k in zip(graphArr.mid_point_names,graphArr.mid_point_x,graphArr.mid_point_y):
        mid_pos_dict[i].append(j)
        mid_pos_dict[i].append(k)

    for i,j,k in zip(nodeArr.NAME,nodeArr.NodeXCoordinate,nodeArr.NodeYCoordinate):
        pos_dict[i].append(j)
        pos_dict[i].append(k)
    pos_dict0 = dict(pos_dict)

    mid_post_dict0=dict(mid_pos_dict)


    temp_0=[0]*graphArr.NAME.size


    node_dict0 = {val:item for val, item in zip(nodeArr.NAME,graphArr.FlowDeviation)}
    edgeflow_dict0 = {val:item for val, item in zip(graphArr.mid_point_names, graphArr.FlowDeviation)}

    d3pos_dict = defaultdict(list)
    midpos_dict = defaultdict(list)

    for d in (mid_post_dict0, edgeflow_dict0): # you can list as many input dicts as you want here
        for key, value in d.items():
            midpos_dict[key].append(value)
    midpos_dict0 = dict(midpos_dict)

    for d in (pos_dict0, node_dict0): # you can list as many input dicts as you want here
        for key, value in d.items():
            d3pos_dict[key].append(value)
    d3pos_dict0 = dict(d3pos_dict)


    edge_list = list(graphArr.mid_point_names)
    node_list = list(nodeArr.NAME)
    temp = list(pos_dict0.keys())
    temp2 = list(mid_post_dict0.keys())
    keys_all=np.append(temp,temp2)

    G.add_nodes_from(keys_all)

    for n in node_list:
        G.nodes[n]['pos'] = pos_dict0[n]
        G.nodes[n]['flow'] = 0

    for n in edge_list:
        G.nodes[n]['pos'] = mid_post_dict0[n]
        G.nodes[n]['flow'] = edgeflow_dict0[n]


    for i in edges:
        pdest = i[1]
        psource = i[0]
        #flow = graphArr['FlowDeviation'].iloc[i]
        name = i[0]
        G.add_edge(psource, pdest, n = name)


    def network_plot_3D(G, angle, save=True):

        lower = min(res0_41['FlowDeviation'])
        upper = max(res0_41['FlowDeviation'])
        colors = plt.cm.jet((graphArr.FlowDeviation-lower)/(upper-lower))


        # 3D network plot
        with plt.style.context(('ggplot')):

            fig = plt.figure(figsize=(30,30))
            ax = Axes3D(fig)
            ax.set_xlabel('x-coordinates',fontsize=30)
            ax.set_ylabel('y-coordinates',fontsize=30)
            ax.set_zlabel('Flow Deviation', fontsize=30)
            #ax.set_zlim(0.0,0.3)


            # Loop on the pos dictionary to extract the x,y,z coordinates of each node
            #for key, value in d3pos_dict0.items():
            #    xi = value[0][0]
            #    yi = value[0][1]
            #    zi = value[1]

                # Scatter plot
                #can add edgecolros
            #    ax.scatter(xi, yi, zi, c='gray', alpha=0.2, s=80)

            ctr=0
            for key, value in midpos_dict0.items():
                xi = value[0][0]
                yi = value[0][1]
                zi = value[1]

                # Scatter plot
                #can add edgecolros
                ax.scatter(xi, yi, zi, c=colors[ctr], alpha=0.7, s=80)
                ctr=ctr+1

            d3pos_dict0.update(midpos_dict0)


            # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
            # Those two points are the extrema of the line to be plotted
            #for i,j in enumerate(G.edges()):
            #    x = np.array((d3pos_dict0[j[0]][0][0], d3pos_dict0[j[1]][0][0]))
            #    y = np.array((d3pos_dict0[j[0]][0][1], d3pos_dict0[j[1]][0][1]))
            #    z = np.array((d3pos_dict0[j[0]][1], d3pos_dict0[j[1]][1]))

            #    ax.plot(x, y, z, c='gray',alpha=0.2)

            for i in edges:
                x = np.array((d3pos_dict0[i[0]][0][0],d3pos_dict0[i[1]][0][0] ))
                y = np.array((d3pos_dict0[i[0]][0][1], d3pos_dict0[i[1]][0][1]))
                z = np.array((d3pos_dict0[i[0]][1], d3pos_dict0[i[1]][1]))

                ax.plot(x, y, z, c='gray',alpha=0.5)

        # Set the initial view
        angleVerticle = 30
        ax.view_init(angleVerticle, angle)
        # Hide the axes
        #ax.set_axis_off()

        if save is not False:
            plt.savefig("/Users/aya/Documents/code-pfs/gas-nx/plots/Leak_41_ZeroDegrees_pipes.png")
            plt.show()
        else:
            plt.show()

        return

    network_plot_3D(G, 60)

def draw_2d(graphArr):
    cntrlnd = '0BEC50B8'
    unique_node=graphArr.FacilityToNodeName.unique()
    u_n_2 = graphArr.FacilityFromNodeName.unique()
    unique_node=np.append(unique_node,u_n_2)
    unique_node=np.unique(unique_node)
    anomalyFreeNode = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Nodes.csv"
    nodeArr=pd.read_csv(anomalyFreeNode)
    nodeArr = nodeArr[nodeArr.NAME.isin(unique_node)]

    G = nx.Graph()

    graphArr['mid_point_names']="mid_point"+graphArr.NAME

    temp_arr = np.array([graphArr.NAME, graphArr.FacilityFromNodeName, graphArr.FacilityToNodeName, graphArr.mid_point_names, graphArr.mid_point_x, graphArr.mid_point_y, graphArr.FlowDeviation,[0]*graphArr.NAME.size,[0]*graphArr.NAME.size])
    edges=np.array([['','',5]])
    m=0
    for i,j in enumerate(temp_arr[3]):
        tmp_src=temp_arr[1][i]
        tmp_dest=temp_arr[2][i]
        srcs = np.where(temp_arr[1]==tmp_src)
        dests = np.where(temp_arr[2] == tmp_src)
        for r in srcs[0]:
            src_node=temp_arr[3][i]
            dest_node=temp_arr[3][r]
            flow=temp_arr[6][i]
            edges = np.vstack([edges, [src_node,dest_node,flow]])
        for k in dests[0]:
            src_node=temp_arr[3][i]
            dest_node=temp_arr[3][k]
            flow=temp_arr[6][i]
            edges = np.vstack([edges, [src_node,dest_node,flow]])
    edges=np.delete(edges,(0), axis=0)

    pos_dict = defaultdict(list)
    mid_pos_dict = defaultdict(list)
    for i, j, k in zip(graphArr.mid_point_names,graphArr.mid_point_x,graphArr.mid_point_y):
        mid_pos_dict[i].append(j)
        mid_pos_dict[i].append(k)

    for i,j,k in zip(nodeArr.NAME,nodeArr.NodeXCoordinate,nodeArr.NodeYCoordinate):
        pos_dict[i].append(j)
        pos_dict[i].append(k)
    pos_dict0 = dict(pos_dict)

    mid_post_dict0=dict(mid_pos_dict)


    temp_0=[0]*graphArr.NAME.size


    node_dict0 = {val:item for val, item in zip(nodeArr.NAME,graphArr.FlowDeviation)}
    edgeflow_dict0 = {val:item for val, item in zip(graphArr.mid_point_names, graphArr.FlowDeviation)}

    d3pos_dict = defaultdict(list)
    midpos_dict = defaultdict(list)

    for d in (mid_post_dict0, edgeflow_dict0): # you can list as many input dicts as you want here
        for key, value in d.items():
            midpos_dict[key].append(value)
    midpos_dict0 = dict(midpos_dict)

    for d in (pos_dict0, node_dict0): # you can list as many input dicts as you want here
        for key, value in d.items():
            d3pos_dict[key].append(value)
    d3pos_dict0 = dict(d3pos_dict)


    edge_list = list(graphArr.mid_point_names)
    node_list = list(nodeArr.NAME)
    temp = list(pos_dict0.keys())
    temp2 = list(mid_post_dict0.keys())
    keys_all=np.append(temp,temp2)

    G.add_nodes_from(edge_list)

    for n in edge_list:
        G.nodes[n]['pos'] = mid_post_dict0[n]
        G.nodes[n]['flow'] = edgeflow_dict0[n]


    for i in edges:
        pdest = i[1]
        psource = i[0]
        flow = i[2]
        name = i[0]
        G.add_edge(psource, pdest,p=flow, n = name)


    n_data = list(G.nodes(data=True))
    p_data = list(G.edges(data=True))

    edgeinfo = nx.get_node_attributes(G, 'flow')

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 30
    fig_size[1] = 30
    plt.rcParams["figure.figsize"] = fig_size
    print("Current size:", fig_size)

    #labels = {}
    #labels[cntrlnd] = r'$\delta$'
    lower = min(res50_41.FlowDeviation)
    upper = max(res50_41.FlowDeviation)


    nodes = G.nodes()
    ec = nx.draw_networkx_edges(G, pos = mid_post_dict0, alpha=1)
    nc = nx.draw_networkx_nodes(G, pos = mid_post_dict0, nodelist=nodes, node_color=graphArr['FlowDeviation'], with_labels=False, node_size=25, cmap=plt.cm.jet, vmin=lower, vmax=upper)
    #lc = nx.draw_networkx_labels(G, pos = mid_post_dict0, labels = labels, font_size=32, font_color='r')

    plt.colorbar(nc)
    plt.axis('off')
    plt.savefig("/Users/aya/Documents/code-pfs/gas-nx/plots/flow2d_pipes_41_zerDeg.png")
    plt.show()

draw_3d(res0_41)

draw_3d(res0_31)

draw_3d(res50_41)

draw_2d(res0_41)

draw_2d(res50_41)

draw_3d(res50_41)
