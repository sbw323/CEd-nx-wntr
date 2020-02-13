
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

#for i, name in enumerate(pipesdf0.columns):
#   print(i+1, name)
#for i, name in enumerate(nodesdf0.columns):
#    print(i+1, name)

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

for i, label in enumerate(pipesdf0['NAME']):
    pdest = pipesdf0['FacilityToNodeName'].iloc[i]
    psource = pipesdf0['FacilityFromNodeName'].iloc[i]
    pressure = pipesdf0['PipeAvePressure'].iloc[i]
    name = pipesdf0['NAME'].iloc[i]
    G.add_edge(psource, pdest, p = pressure, n = name)

n_data = list(G.nodes(data=True))
p_data = list(G.edges(data=True))

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
ec = nx.draw_networkx_edges(G, pos = pos_dict0, alpha=1)
nc = nx.draw_networkx_nodes(G, pos = pos_dict0, nodelist=nodes, node_color=nodesdf0['NodePressure'], with_labels=False, node_size=25, cmap=plt.cm.jet)
lc = nx.draw_networkx_labels(G, pos = pos_dict0, labels = labels, font_size=32, font_color='r')

plt.colorbar(nc)
plt.axis('off')
#plt.savefig("/Users/aya/Documents/code-pfs/gas-nx/plots/ZeroDegrees.png")
plt.show()
