import pandas as pd
from pandas import DataFrame as df
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import wntr as wn
import networkx as nx
from collections import defaultdict
import tensorflow as tf

anomalyFree = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Pipes.csv"
nFile0=pd.read_csv(anomalyFree)

anomaly = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Pipes_Leak1.csv"
nFile1=pd.read_csv(anomaly)

def get_file(name):
    anomaly = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/"+name
    nFile=pd.read_csv(anomaly)
    return nFile

def flowDeviation(file0, file1):
    res_arr = file1
    res_arr['FlowDeviation']= file1.FacilityFlowAbsolute
    res_arr.FlowDeviation = abs(res_arr.FacilityFlowAbsolute.subtract(file0.FacilityFlowAbsolute))/file0.FacilityFlowAbsolute
    return res_arr

def flowDeviationNormalized(res_arr):
    norm_arr = res_arr
    norm_arr['NormalizedDeviation']= norm_arr.FlowDeviation
    lower = min(norm_arr.FlowDeviation)
    upper = max(norm_arr.FlowDeviation)
    norm_arr.FlowDeviation = ((norm_arr.FlowDeviation-lower)/(upper-lower))
    return norm_arr

preDir = "/LeakData_ZeroDegrees/"
name0_11="NYU Anamoly Data_ZeroDeg_Pipes_Leak11.csv"
name0_21="NYU Anamoly Data_ZeroDeg_Pipes_Leak21.csv"
name0_31="NYU Anamoly Data_ZeroDeg_Pipes_Leak31.csv"
name0_41="NYU Anamoly Data_ZeroDeg_Pipes_Leak41.csv"

leak0_11 = get_file(preDir+name0_11)
leak0_21 = get_file(preDir+name0_21)
leak0_31 = get_file(preDir+name0_31)
leak0_41 = get_file(preDir+name0_41)

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

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 30
fig_size[1] = 30
plt.rcParams["figure.figsize"] = fig_size
print("Current size:", fig_size)

def draw_graph(nodeArr,preshArr):
    cntrlnd = '0BEC50B8'

    G = nx.Graph()

    pos_dict = defaultdict(list)
    for i, j, k in zip(nodeArr.NAME,nodeArr.NodeXCoordinate,nodeArr.NodeYCoordinate):
        pos_dict[i].append(j)
        pos_dict[i].append(k)
    pos_dict0 = dict(pos_dict)

    edge_list = list(preshArr.NAME)
    G.add_nodes_from(pos_dict0.keys())

    for i, label in enumerate(preshArr['NAME']):
        pdest = preshArr['FacilityToNodeName'].iloc[i]
        psource = preshArr['FacilityFromNodeName'].iloc[i]
        flow = preshArr['NormalizedDeviation'].iloc[i]
        name = preshArr['NAME'].iloc[i]
        G.add_edge(psource, pdest, p = flow, n = name)

    n_data = list(G.nodes(data=True))
    p_data = list(G.edges(data=True))

    edgeinfo = nx.get_edge_attributes(G, 'flow')

    labels = {}
    labels[cntrlnd] = r'$\delta$'

    nodes = G.nodes()
    edges = G.edges()
    lower = min(norm_41.FlowDeviation)
    upper = max(norm_41.FlowDeviation)
    colors = plt.cm.jet(norm_41['FlowDeviation'])

    ec = nx.draw_networkx_edges(G, pos = pos_dict0, colors = norm_41['FlowDeviation'], width=4, cmap=plt.cm.Blues, with_labels=False, edge_vmin=lower, edge_vmax=upper)
    nc = nx.draw_networkx_nodes(G, pos = pos_dict0, alpha=1, node_size=25)
    lc = nx.draw_networkx_labels(G, pos = pos_dict0, labels = labels, font_size=32, font_color='r')

    plt.colorbar(ec)
    plt.axis('off')
    #plt.savefig("/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/press5.png")
    plt.show()

anomalyFreeNode = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Nodes.csv"
norm_41 = flowDeviationNormalized(res0_41)
nodeArr=pd.read_csv(anomalyFreeNode)
draw_graph(nodeArr,res0_41)

from mpl_toolkits.mplot3d import Axes3D

def draw_3d(nodeArr,preshArr):

    cntrlnd = '0BEC50B8'

    G = nx.Graph()

    pos_dict = defaultdict(list)
    for i, j, k in zip(nodeArr.NAME,nodeArr.NodeXCoordinate,nodeArr.NodeYCoordinate):
        pos_dict[i].append(j)
        pos_dict[i].append(k)
    pos_dict0 = dict(pos_dict)

    #nodepressure_dict0 = {val:item for val, item in zip(graphArr.NAME,graphArr.PressureDeviation)}
    edgeflow_dict0 = {val:item for val, item in zip(preshArr.NAME, preshArr.FlowDeviation)}

    d3pos_dict = defaultdict(list)

    for d in (pos_dict0, edgeflow_dict0): # you can list as many input dicts as you want here
        for key, value in d.items():
            d3pos_dict[key].append(value)
    d3pos_dict0 = dict(d3pos_dict)

    node_list = list(nodeArr.NAME)
    edge_list = list(preshArr.NAME)
    G.add_nodes_from(pos_dict0.keys())
    for n in edge_list:
        G.nodes[n]['pos'] = pos_dict0[n]
        G.nodes[n]['flow'] = edgeflow_dict0[n]

    for i, label in enumerate(preshArr['NAME']):
        pdest = preshArr['FacilityToNodeName'].iloc[i]
        psource = preshArr['FacilityFromNodeName'].iloc[i]
        flow = preshArr['FlowDeviation'].iloc[i]
        name = preshArr['NAME'].iloc[i]
        G.add_edge(psource, pdest, p = flow, n = name)

    def network_plot_3D(G, angle, save=True):

        lower = min(preshArr['FlowDeviation'])
        upper = max(preshArr['FlowDeviation'])
        colors = plt.cm.jet((preshArr.FlowDeviation-lower)/(upper-lower))


        # 3D network plot
        with plt.style.context(('ggplot')):

            fig = plt.figure(figsize=(30,30))
            ax = Axes3D(fig)
            ax.set_xlabel('x-coordinates',fontsize=30)
            ax.set_ylabel('y-coordinates',fontsize=30)
            ax.set_zlabel('Pressure Deviation', fontsize=30)
            #ax.set_zlim(0.0,0.3)


            # Loop on the pos dictionary to extract the x,y,z coordinates of each node
            ctr=0
            for key, value in d3pos_dict0.items():
                xi = value[0][0]
                yi = value[0][1]
                zi = value[1]

                # Scatter plot
                ax.scatter(xi, yi, zi, edgecolors=colors[ctr], alpha=0.5, s=80)
                ctr=ctr+1

            # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
            # Those two points are the extrema of the line to be plotted
            for i,j in enumerate(G.edges()):
                x = np.array((d3pos_dict0[j[0]][0][0], d3pos_dict0[j[1]][0][0]))
                y = np.array((d3pos_dict0[j[0]][0][1], d3pos_dict0[j[1]][0][1]))
                z = np.array((d3pos_dict0[j[0]][1], d3pos_dict0[j[1]][1]))

            # Plot the connecting lines
                ax.plot(x, y, z, c='black',alpha=0.5)

        # Set the initial view
        angleVerticle = 30
        ax.view_init(angleVerticle, angle)
        # Hide the axes
        #ax.set_axis_off()

        if save is not False:
            plt.savefig("/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/Leak_11_ZeroDegrees.png")
            plt.show()
        else:
            plt.show()

        return

    network_plot_3D(G, 60)

max(res0_1.FlowDeviation)

max(res0_11.FlowDeviation)

max(res0_21.FlowDeviation)

max(res0_31.FlowDeviation)

max(res0_41.FlowDeviation)

from sklearn import svm
svc = svm.SVC(probability=False,  kernel="linear", C=2.8, gamma=.0073,verbose=10)

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

anomalyFreeNode = "/Users/kavyaub/Documents/mySubjects/ConEdison/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Nodes.csv"
nodeArr=pd.read_csv(anomalyFreeNode)
setOfNames = set(nodeArr['NAME'])
res0_41["FacilityFromNodeNameXCoord"]=0.0
res0_41["FacilityFromNodeNameYCoord"]=0.0
for row in res0_41.iterrows():
    if row[1]['FacilityFromNodeName'] in setOfNames:
        row1 = nodeArr.loc[nodeArr['NAME']==row[1]['FacilityFromNodeName']]
        row.at[1,"FacilityFromNodeNameXCoord"]=row1.iloc[0][3]
        row.at[1,"FacilityFromNodeNameYCoord"]=row1.iloc[0][2]
    elif row[1]['FacilityToNodeName'] in setOfNames:
        row1 = nodeArr.loc[nodeArr['NAME']==row[1]['FacilityToNodeName']]
        row[1]["FacilityFromNodeNameXCoord"]=row1['NodeXCoordinate']
        row[1]["FacilityFromNodeNameYCoord"]=row1['NodeYCoordinate']

max(res0_41["FacilityFromNodeNameXCoord"])

final_res0_41 = np.array([res0_41['NAME'],res0_41['FacilityFromNodeName'],res0_41['FacilityToNodeName'],res0_41['FlowDeviation'],[0]*res0_41.NAME.size,[0]*res0_41.NAME.size,[0]*res0_41.NAME.size,[0]*res0_41.NAME.size])

anomalyFreeNode = "/Users/kavyaub/Documents/mySubjects/ConEdison/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Nodes.csv"
nodeArr=pd.read_csv(anomalyFreeNode)
setOfNames = set(nodeArr['NAME'])

for i in range(0,final_res0_41[1].size):
    if final_res0_41[1][i] in setOfNames:
        temp = nodeArr.loc[nodeArr['NAME']==final_res0_41[1][i]]
        final_res0_41[4][i]=temp.iloc[0][3]
        final_res0_41[5][i]=temp.iloc[0][2]
    if final_res0_41[2][i] in setOfNames:
        temp = nodeArr.loc[nodeArr['NAME']==final_res0_41[2][i]]
        final_res0_41[6][i]=temp.iloc[0][3]
        final_res0_41[7][i]=temp.iloc[0][2]

#final_res0_41['NAME']=res0_41['NAME']
#final_res0_41['FacilityFromNodeName']=res0_41['FacilityFromNodeName']
#final_res0_41['FacilityToNodeName']=res0_41['FacilityToNodeName']
#final_res0_41['FlowDeviation']=res0_41['FlowDeviation']
#final_res0_41['FacilityFromNodeNameXCoord']=0.0
#final_res0_41['FacilityFromNodeNameYCoord']=0.0
#final_res0_41['FacilityToNodeNameXCoord']=0.0
#final_res0_41['FacilityToNodeNameYCoord']=0.0
draw_3d(nodeArr,res0_41)

def draw_graph_np(graphArr):
    cntrlnd = '0BEC50B8'

    G = nx.Graph()

    pos_dict = defaultdict(list)
    for i, j, k in zip(graphArr[0],graphArr[4],graphArr[5]):
        pos_dict[i].append(j)
        pos_dict[i].append(k)
    pos_dict0 = dict(pos_dict)

    #nodepressure_dict0 = {val:item for val, item in zip(nodeArr.NAME,nodeArr.FlowDeviation)}

    #node_list = list(nodeArr.NAME)
    edge_list = list(graphArr[0])
    G.add_nodes_from(pos_dict0.keys())
    #for n in node_list:
    #    G.nodes[n]['pos'] = pos_dict0[n]
    #    G.nodes[n]['flow'] = nodepressure_dict0[n]

    for i in range(0,graphArr[0].size):
        pdest = graphArr[2][i]
        psource = graphArr[1][i]
        flow = graphArr[3][i]
        name = graphArr[0][i]
        G.add_edge(psource, pdest, p = flow, n = name)

    n_data = list(G.nodes(data=True))
    p_data = list(G.edges(data=True))

    #nodeinfo = nx.get_node_attributes(G, 'pressure')
    edgeinfo = nx.get_edge_attributes(G, 'flow')
    #nodeinfo[cntrlnd]

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 30
    fig_size[1] = 30
    plt.rcParams["figure.figsize"] = fig_size
    print("Current size:", fig_size)

    labels = {}
    labels[cntrlnd] = r'$\delta$'

    nodes = G.nodes()
    edges = G.edges()
    lower = min(graphArr[3])
    upper = max(graphArr[3])
    #colors = plt.cm.jet((graphArr[3]-lower)/(upper-lower))
    #colors=range(round(upper))

    #edges = nx.draw_networkx_edges(G,pos,edge_color=colors,width=4,
    #                           edge_cmap=plt.cm.Blues)
    #print(nodeArr.shape)
    #print(colors.shape)
    print(pos_dict0)
    ec = nx.draw_networkx_edges(G, pos = pos_dict0, edge_color=graphArr[3], with_labels=False,cmap = plt.cm.jet)
    nc = nx.draw_networkx_nodes(G, pos = pos_dict0, alpha=1)
    lc = nx.draw_networkx_labels(G, pos = pos_dict0, labels = labels, font_size=32, font_color='r')

    plt.colorbar(ec)
    plt.axis('off')
    #plt.savefig("/Users/kavyaub/Documents/mySubjects/ConEdison/screenshots/press5.png")
    plt.show()

draw_graph_np(final_res0_41)
