
import networkx as nx
import pandas as pd
from pandas import DataFrame as df
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

pnodes = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Nodes.csv"
ppipes = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Pipes.csv"
pipesdf0 = pd.read_csv(ppipes)
nodesdf0 = pd.read_csv(pnodes)
pipesdf0.dropna(axis = 1, how = 'all', inplace = True)
nodesdf0.dropna(axis = 1, how = 'all', inplace = True)
cntrlnd = '0BEC50B8'

for i, name in enumerate(pipesdf0.columns):
   print(i+1, name)
#for i, name in enumerate(nodesdf0.columns):
#    print(i+1, name)

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

norm_41 = flowDeviationNormalized(res0_41)

for i, name in enumerate(norm_41.columns):
   print(i+1, name)

G = nx.Graph()

pos_dict = defaultdict(list)
for i, j, k in zip(nodesdf0.NAME,nodesdf0.NodeXCoordinate,nodesdf0.NodeYCoordinate):
    pos_dict[i].append(j)
    pos_dict[i].append(k)
pos_dict0 = dict(pos_dict)

nodepressure_dict0 = {val:item for val, item in zip(nodesdf0.NAME,nodesdf0.NodePressure)}

node_list = list(nodesdf0.NAME)
G.add_nodes_from(pos_dict0.keys())
for n in node_list:
    G.node[n]['pos'] = pos_dict0[n]
    G.node[n]['pressure'] = nodepressure_dict0[n]

for i, label in enumerate(norm_41['NAME']):
    pdest = norm_41['FacilityToNodeName'].iloc[i]
    psource = norm_41['FacilityFromNodeName'].iloc[i]
    pressure = norm_41['PipeAvePressure'].iloc[i]
    normFlowdv = norm_41['NormalizedDeviation'].iloc[i]
    name = norm_41['NAME'].iloc[i]
    G.add_edge(psource, pdest, p = pressure, n = name, f = normFlowdv)

n_data = list(G.nodes(data=True))
p_data = list(G.edges(data=True))
p_data[0:2]

nodeinfo = nx.get_node_attributes(G, 'pressure')
nodeinfo[cntrlnd]

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 30
fig_size[1] = 30
plt.rcParams["figure.figsize"] = fig_size
print("Current size:", fig_size)

labels = {}
labels[cntrlnd] = r'$\delta$'

nodes = G.nodes()
ec = nx.draw_networkx_edges(G, pos = pos_dict0, edge_color = pipesdf0['FacilityFlowAbsolute'], alpha=1)
nc = nx.draw_networkx_nodes(G, pos = pos_dict0, nodelist=nodes, node_color=nodesdf0['NodePressure'], with_labels=False, node_size=25, cmap=plt.cm.jet)
lc = nx.draw_networkx_labels(G, pos = pos_dict0, labels = labels, font_size=32, font_color='r')

plt.colorbar(nc)
plt.axis('off')
#plt.savefig("/Users/aya/Documents/code-pfs/gas-nx/plots/ZeroDegrees.png")
plt.show()
len(pipesdf0['FacilityFlowAbsolute'])
