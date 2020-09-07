#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from pandas import DataFrame as df
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wntr as wn
import networkx as nx
from collections import defaultdict
import tensorflow as tf


# In[13]:


anomalyFree = "/Users/kavyaub/Documents/mySubjects/ConEdison/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Pipes.csv"
nFile0=pd.read_csv(anomalyFree)


# In[14]:


anomaly = "/Users/kavyaub/Documents/mySubjects/ConEdison/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Pipes_Leak1.csv"
nFile1=pd.read_csv(anomaly)


# In[15]:


def get_file(name):
    anomaly = "/Users/kavyaub/Documents/mySubjects/ConEdison/NYU_LeakData"+name
    nFile=pd.read_csv(anomaly)
    return nFile


# In[56]:


def flowDeviation_vel(file0, file1):
    res_arr = file1
    res_arr['VELOpipeFPS']= file1.VELOpipeFPS
    res_arr.VELOpipeFPS = abs(res_arr.VELOpipeFPS.subtract(file0.VELOpipeFPS))
    return res_arr


# In[57]:


import math
in2_ft2 = 0.0005787
setOfNames = set(leak0_41['NAME'])
def pipe_velocity_calc(pipedf):
    final_pipedf = np.array([pipedf['NAME'],pipedf['FacilityFromNodeName'],pipedf['FacilityToNodeName'],pipedf['FacilityFlowAbsolute'],pipedf['PipeDiameter'],[0]*pipedf.NAME.size,[0]*pipedf.NAME.size,[0]*pipedf.NAME.size,[0]*pipedf.NAME.size,[0]*pipedf.NAME.size,[0]*pipedf.NAME.size])

    for i in range(0,final_pipedf[1].size):
        if final_pipedf[0][i] in setOfNames:
            temp = pipedf.loc[pipedf['NAME']==final_pipedf[0][i]]
            final_pipedf[5][i]=temp.iloc[0][90]
            final_pipedf[6][i]=temp.iloc[0][94]

    ## Creating the Velocity Rate Column ##
    elem_AREApipeFT2 = (final_pipedf[6]**2/4*math.pi*in2_ft2)
    elem_VELOpipeFPS = (final_pipedf[5]/elem_AREApipeFT2*1000/3600)
    final_pipedf[8] = elem_AREApipeFT2
    final_pipedf[9] = elem_VELOpipeFPS
    pipedf['AREApipeFT2'] = elem_AREApipeFT2
    pipedf['VELOpipeFPS'] = elem_VELOpipeFPS

    return pipedf


# In[58]:


def calculations(input_arr):
    final_temp_arr = np.array([input_arr['NAME'],input_arr['FacilityFromNodeName'],input_arr['FacilityToNodeName'],input_arr['NAME'],[0]*input_arr.NAME.size,[0]*input_arr.NAME.size,[0]*input_arr.NAME.size,[0]*input_arr.NAME.size,[0]*input_arr.NAME.size,[0]*input_arr.NAME.size])

    anomalyFreeNode = "/Users/kavyaub/Documents/mySubjects/ConEdison/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Nodes.csv"
    nodeArr=pd.read_csv(anomalyFreeNode)
    setOfNames = set(nodeArr['NAME'])

    for i in range(0,final_temp_arr[1].size):
        if final_temp_arr[1][i] in setOfNames:
            temp = nodeArr.loc[nodeArr['NAME']==final_temp_arr[1][i]]
            final_temp_arr[4][i]=temp.iloc[0][3]
            final_temp_arr[5][i]=temp.iloc[0][2]
        if final_temp_arr[2][i] in setOfNames:
            temp = nodeArr.loc[nodeArr['NAME']==final_temp_arr[2][i]]
            final_temp_arr[6][i]=temp.iloc[0][3]
            final_temp_arr[7][i]=temp.iloc[0][2]
            
    mid_elem_x=(final_temp_arr[4]+final_temp_arr[6])/2.0
    mid_elem_y=(final_temp_arr[5]+final_temp_arr[7])/2.0
    
    final_temp_arr[8]=mid_elem_x
    final_temp_arr[9]=mid_elem_y
    input_arr['mid_point_x']=mid_elem_x
    input_arr['mid_point_y']=mid_elem_y


# In[59]:


preDir = "/LeakData_ZeroDegrees/"
name0_11="NYU Anamoly Data_ZeroDeg_Pipes_Leak11.csv"
name0_21="NYU Anamoly Data_ZeroDeg_Pipes_Leak21.csv"
name0_31="NYU Anamoly Data_ZeroDeg_Pipes_Leak31.csv"
name0_41="NYU Anamoly Data_ZeroDeg_Pipes_Leak41.csv"

leak0_11 = get_file(preDir+name0_11)
leak0_21 = get_file(preDir+name0_21)
leak0_31 = get_file(preDir+name0_31)
leak0_41 = get_file(preDir+name0_41)


# In[60]:


res0_1 = pipe_velocity_calc(nFile1)
normal_vel_1 = pipe_velocity_calc(nFile0)
cal_diff_1 = flowDeviation_vel(normal_vel_1,res0_1)
calculations(cal_diff_1)

res0_11 = pipe_velocity_calc(leak0_11)
normal_vel_11 = pipe_velocity_calc(nFile0)
cal_diff_11 = flowDeviation_vel(normal_vel_11,res0_11)
calculations(cal_diff_11)

res0_21 = pipe_velocity_calc(leak0_21)
normal_vel_21 = pipe_velocity_calc(nFile0)
cal_diff_21 = flowDeviation_vel(normal_vel_21,res0_21)
calculations(cal_diff_21)

res0_31 = pipe_velocity_calc(leak0_31)
normal_vel_31 = pipe_velocity_calc(nFile0)
cal_diff_31 = flowDeviation_vel(normal_vel_31,res0_31)
calculations(cal_diff_31)

res0_41 = pipe_velocity_calc(leak0_41)
normal_vel_41 = pipe_velocity_calc(nFile0)
cal_diff = flowDeviation_vel(normal_vel_41,res0_41)
calculations(cal_diff)


# In[ ]:





# In[61]:


cal_diff_1 = cal_diff_1.fillna(value=0.0)
cal_diff_11 = cal_diff_11.fillna(value=0.0)
cal_diff_21 = cal_diff_21.fillna(value=0.0)
cal_diff_31 = cal_diff_31.fillna(value=0.0)
cal_diff = cal_diff.fillna(value=0.0)


# In[62]:


cal_diff_1.to_csv(r'/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/pipes1.csv')
cal_diff_11.to_csv(r'/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/pipes11.csv')
cal_diff_21.to_csv(r'/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/pipes21.csv')
cal_diff_31.to_csv(r'/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/pipes31.csv')
cal_diff.to_csv(r'/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/pipes41.csv')


# In[ ]:





# In[63]:


cal_diff_1['leak'] = np.where(cal_diff_1['VELOpipeFPS']>=40, 1, 0)
cal_diff_11['leak'] = np.where(cal_diff_11['VELOpipeFPS']>=40, 1, 0)
cal_diff_21['leak'] = np.where(cal_diff_21['VELOpipeFPS']>=40, 1, 0)
cal_diff_31['leak'] = np.where(cal_diff_31['VELOpipeFPS']>=40, 1, 0)
cal_diff['leak'] = np.where(cal_diff['VELOpipeFPS']>=40, 1, 0)


# In[64]:


leakTable=pd.DataFrame(columns=['leak1','leak2','leak3','leak4','leak5'])
leakTable['leak1']=cal_diff_1['leak']
leakTable['leak2']=cal_diff_11['leak']
leakTable['leak3']=cal_diff_21['leak']
leakTable['leak4']=cal_diff_31['leak']
leakTable['leak5']=cal_diff['leak']
leakTable['sumLeaks']=leakTable.leak1+leakTable.leak2+leakTable.leak3+leakTable.leak4+leakTable.leak5


# In[65]:


arr=range(cal_diff_1.VELOpipeFPS.size)
plt.plot(arr,leakTable.sumLeaks)


# In[66]:


leakTable.sumLeaks.value_counts()


# In[67]:


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


# In[25]:


leakTable.style.apply(hightlightColor, axis=1)


# In[68]:


anomalyFreeNode = "/Users/kavyaub/Documents/mySubjects/ConEdison/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Nodes.csv"
nodeArr=pd.read_csv(anomalyFreeNode)


# In[69]:


draw_3d(cal_diff)


# In[160]:


from mpl_toolkits.mplot3d import Axes3D

def draw_3d(graphArr):
    cntrlnd = '0BEC50B8'
    unique_node=graphArr.FacilityToNodeName.unique()
    u_n_2 = graphArr.FacilityFromNodeName.unique()
    unique_node=np.append(unique_node,u_n_2)
    unique_node=np.unique(unique_node)
    anomalyFreeNode = "/Users/kavyaub/Documents/mySubjects/ConEdison/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Nodes.csv"
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


    node_dict0 = {val:item for val, item in zip(nodeArr.NAME,graphArr.VELOpipeFPS)}
    edgeflow_dict0 = {val:item for val, item in zip(graphArr.mid_point_names, graphArr.VELOpipeFPS)}

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
        name = i[0]
        G.add_edge(psource, pdest, n = name)


    def network_plot_3D(G, angle, save=True):
    
        lower = min(cal_diff['VELOpipeFPS'])
        upper = max(cal_diff['VELOpipeFPS'])
        tp_arr = (graphArr['VELOpipeFPS']-lower)/(upper-lower)
        tp2 = [float(i) for i in tp_arr]
        
        colors = ["#1a1aff", "#00cc00", "#ffff00", "#ffa500", "#ff4d4d"]
        cmap= matplotlib.colors.ListedColormap(colors)
        cmap.set_under("crimson")
        cmap.set_over("w")
        norm= matplotlib.colors.Normalize(vmin=lower,vmax=upper)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cc = cmap(tp2)


        # 3D network plot
        with plt.style.context(('ggplot')):

            fig = plt.figure(figsize=(30,30))
            ax = Axes3D(fig)
            ax.set_xlabel('x-coordinates',fontsize=30)
            ax.set_ylabel('y-coordinates',fontsize=30)
            ax.set_zlabel('Flow Deviation', fontsize=30)
            ax.set_zlim(lower,upper)
            
            ctr=0
            for key, value in midpos_dict0.items():
                xi = value[0][0]
                yi = value[0][1]
                zi = value[1]

                p = ax.scatter(xi, yi, zi, c=cc[ctr], alpha=0.7, s=80,cmap=cmap)
                ctr=ctr+1
            d3pos_dict0.update(midpos_dict0)

        
            for i in edges:
                x = np.array((d3pos_dict0[i[0]][0][0],d3pos_dict0[i[1]][0][0] ))
                y = np.array((d3pos_dict0[i[0]][0][1], d3pos_dict0[i[1]][0][1]))
                z = np.array((d3pos_dict0[i[0]][1], d3pos_dict0[i[1]][1]))

                ax.plot(x, y, z, c='gray',alpha=0.5)
            #fig.colorbar(p)


        # Set the initial view
        angleVerticle = 30
        ax.view_init(angleVerticle, angle)
        

        if save is not False:
            plt.savefig("/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/Leak_41_ZeroDegrees_pipes.png")
            plt.show()
        else:
            plt.show()

        return

    network_plot_3D(G, 60)


# In[161]:


draw_3d(cal_diff)


# In[162]:


draw_3d(cal_diff_11)


# In[ ]:


draw_3d(cal_diff_21)


# In[ ]:


draw_3d(cal_diff_31)


# In[ ]:


def draw_2d_flow(graphArr):
    cntrlnd = '0BEC50B8'
    unique_node=graphArr.FacilityToNodeName.unique()
    u_n_2 = graphArr.FacilityFromNodeName.unique()
    unique_node=np.append(unique_node,u_n_2)
    unique_node=np.unique(unique_node)
    anomalyFreeNode = "/Users/kavyaub/Documents/mySubjects/ConEdison/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Nodes.csv"
    nodeArr=pd.read_csv(anomalyFreeNode)
    nodeArr = nodeArr[nodeArr.NAME.isin(unique_node)]

    G = nx.Graph()

    graphArr['mid_point_names']="mid_point"+graphArr.NAME

    temp_arr = np.array([graphArr.NAME, graphArr.FacilityFromNodeName, graphArr.FacilityToNodeName, graphArr.mid_point_names, graphArr.mid_point_x, graphArr.mid_point_y, [0]*graphArr.NAME.size,[0]*graphArr.NAME.size,[0]*graphArr.NAME.size])
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


    node_dict0 = {val:item for val, item in zip(nodeArr.NAME,graphArr.VELOpipeFPS)}
    edgeflow_dict0 = {val:item for val, item in zip(graphArr.mid_point_names, graphArr.VELOpipeFPS)}

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

    labels = {}
    labels['mid_point0BF5C083'] = r'$\delta$'
    labels['mid_point0BF5C2D7'] = r'$\delta$'
    lower = min(cal_diff.VELOpipeFPS)
    upper = max(cal_diff.VELOpipeFPS)
    
    colors = ["#1a1aff", "#00cc00", "#ffff00", "#ffa500", "#ff4d4d"]
    cmap= matplotlib.colors.ListedColormap(colors)
    cmap.set_under("crimson")
    cmap.set_over("w")
    norm= matplotlib.colors.Normalize(vmin=lower,vmax=upper)

    nodes = G.nodes()
    ec = nx.draw_networkx_edges(G, pos = mid_post_dict0, alpha=1)
    nc = nx.draw_networkx_nodes(G, pos = mid_post_dict0, nodelist=nodes, node_color=graphArr['VELOpipeFPS'], with_labels=False, node_size=25, cmap=cmap, vmin=lower, vmax=upper)
    lc = nx.draw_networkx_labels(G, pos = mid_post_dict0, labels = labels, font_size=32, font_color='r')
    
    plt.colorbar(nc)
    plt.axis('off')
    plt.savefig("/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/flow2d_pipes_41_zerDeg.png")
    plt.show()


# In[ ]:


draw_2d_flow(cal_diff)


# In[ ]:


draw_2d_flow(cal_diff_11)


# In[ ]:


draw_2d_flow(cal_diff_21)


# In[ ]:


draw_2d_flow(cal_diff_31)


# In[ ]:


draw_2d_flow(cal_diff)


# In[76]:




