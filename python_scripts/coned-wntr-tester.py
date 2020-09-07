import wntr
import networkx as nx
import pandas as pd
from pandas import DataFrame as df
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

pnodes = "/Users/aya/Documents/NYU/progressfolders/10152019/NYU2-nodes.xls"
ppipes = "/Users/aya/Documents/NYU/progressfolders/10152019/NYU-pipes.xls"
pregulators = "/Users/aya/Documents/NYU/progressfolders/10152019/NYU3-regulators.xls"
pipesdf0 = pd.read_excel(ppipes, sheet_name='Sheet1')
nodesdf0 = pd.read_excel(pnodes, sheet_name='Sheet1')
regulatorsdf = pd.read_excel(pregulators, sheet_name='Sheet1')
pipesdf0.dropna(axis = 1, how = 'all', inplace = True)
nodesdf0.dropna(axis = 1, how = 'all', inplace = True)
regulatorsdf.dropna(axis = 1, how = 'all', inplace = True)

for i, name in enumerate(pipesdf0.columns):
    print(i+1, name)
for i, name in enumerate(nodesdf0.columns):
    print(i+1, name)
for i, name in enumerate(regulatorsdf.columns):
    print(i+1, name)

pos_dict = defaultdict(list)
for i, j, k in zip(nodesdf0.NAME,nodesdf0.NodeXCoordinate,nodesdf0.NodeYCoordinate):
    pos_dict[i].append(j)
    pos_dict[i].append(k)
pos_dict0 = dict(pos_dict)

wn = wntr.network.WaterNetworkModel()

node_list = list(nodesdf0.NAME)
for i in node_list:
    wn.add_junction(name = i, base_demand=10, demand_pattern='1', elevation=0, coordinates=pos_dict0[i])

for i, label in enumerate(pipesdf0['NAME']):
    pname = label
    pdest = pipesdf0['FacilityToNodeName'].iloc[i]
    psource = pipesdf0['FacilityFromNodeName'].iloc[i]
    plen = pipesdf0['PipeLength'].iloc[i]
    pdia = pipesdf0['PipeDiameter'].iloc[i]
    prough = pipesdf0['PipeRoughness'].iloc[i]
    wn.add_pipe(name = pname, start_node_name=psource, end_node_name=pdest, length=plen, diameter=pdia, roughness=prough, minor_loss=0)

G = wn.get_graph()
hangingnodes = list(nx.isolates(G))
pipecheck_tolist = []
pipecheck_tolist.extend(pipesdf0['FacilityToNodeName'])
pipecheck_fromlist = []
pipecheck_fromlist.extend(pipesdf0['FacilityFromNodeName'])
to_temp = [x for x in hangingnodes if x in pipecheck_tolist]
from_temp = [x for x in hangingnodes if x in pipecheck_fromlist]
if len(to_temp) == 0:
    print("error:no pipes connecting to regulators")
if len(from_temp) == 0:
    print("error:no pipes connecting from regulators")

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 30
fig_size[1] = 30
plt.rcParams["figure.figsize"] = fig_size
print("Current size:", fig_size)

wntr.graphics.plot_network(wn)
