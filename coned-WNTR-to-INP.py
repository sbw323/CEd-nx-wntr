import wntr
import networkx as nx
import pandas as pd
from pandas import DataFrame
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import math

all_nodes = "/Users/aya/Documents/NYU/progressfolders/10152019/NYU2-nodes.xls"
pnodes = "/Users/aya/Documents/NYU/progressfolders/11172019/selectednodes0degrees_export_TableToExcel.xls"
ppipes = "/Users/aya/Documents/NYU/progressfolders/10152019/NYU-pipes.xls"
pregulators = "/Users/aya/Documents/NYU/progressfolders/10152019/NYU3-regulators.xls"
all_nodes = pd.read_excel(all_nodes, sheet_name='Sheet1')
pipesdf0 = pd.read_excel(ppipes, sheet_name='Sheet1')
nodesdf0 = pd.read_excel(pnodes, sheet_name='Sheet1')
regulatorsdf = pd.read_excel(pregulators, sheet_name='Sheet1')
all_nodes.dropna(axis = 1, how = 'all', inplace = True)
pipesdf0.dropna(axis = 1, how = 'all', inplace = True)
nodesdf0.dropna(axis = 1, how = 'all', inplace = True)
regulatorsdf.dropna(axis = 1, how = 'all', inplace = True)

for i, name in enumerate(pipesdf0.columns):
    print(i+1, name)
for i, name in enumerate(nodesdf0.columns):
    print(i+1, name)
for i, name in enumerate(regulatorsdf.columns):
    print(i+1, name)

selectednodes = list(nodesdf0['NAME'])
pipesfrom_df = pipesdf0[pipesdf0['FacilityToNodeName'].isin(selectednodes)]
pipesto_df = pipesdf0[pipesdf0['FacilityFromNodeName'].isin(selectednodes)]
resultpipes_df = pd.concat([pipesto_df, pipesfrom_df], ignore_index = True)
resultspipes_nodupes_df = resultpipes_df.drop_duplicates(subset = 'NAME', inplace = False)
#resultpipes_df.dropna(axis = 1, how = 'all', inplace = True)
selectedpipes_to = list(resultpipes_df['FacilityToNodeName'])
selectedpipes_from = list(resultpipes_df['FacilityFromNodeName'])
revised_nodes_to_df = all_nodes[all_nodes['NAME'].isin(selectedpipes_to)]
revised_nodes_from_df = all_nodes[all_nodes['NAME'].isin(selectedpipes_from)]
revised_nodes_df = pd.concat([revised_nodes_to_df, revised_nodes_from_df], ignore_index = True)
revised_nodes_df.drop_duplicates(subset = 'NAME', inplace = True)
revised_nodes_df.dropna(axis = 1, how = 'all', inplace = True)
print("shape of chosen nodes df is: ",revised_nodes_df.shape)
print("shape of chosen pipe df is: ", resultpipes_df.shape)
print("shape of chosen pipe df w/ duplicates removed is: ",resultspipes_nodupes_df.shape)
#resultpipes_df.to_excel("/Users/aya/Documents/code-pfs/gas-nx/resultpipes_xl.xlsx")
pipesto_df['FacilityToNodeName']
pipesfrom_df['FacilityToNodeName']

pos_dict = defaultdict(list)
for i, j, k in zip(revised_nodes_df.NAME,revised_nodes_df.NodeXCoordinate,revised_nodes_df.NodeYCoordinate):
    pos_dict[i].append(j)
    pos_dict[i].append(k)
pos_dict0 = dict(pos_dict)

wn = wntr.network.WaterNetworkModel()

node_list = list(revised_nodes_df.NAME)
for i in node_list:
    wn.add_junction(name = i, base_demand=1, elevation=0, coordinates=pos_dict0[i])

for i, label in enumerate(resultpipes_df['NAME']):
    pname = label
    pdest = resultpipes_df['FacilityToNodeName'].iloc[i]
    psource = resultpipes_df['FacilityFromNodeName'].iloc[i]
    plen = resultpipes_df['PipeLength'].iloc[i]
    pdia = resultpipes_df['PipeDiameter'].iloc[i]
    prough = resultpipes_df['PipeRoughness'].iloc[i]
    wn.add_pipe(name = pname, start_node_name=psource, end_node_name=pdest, length=plen, diameter=pdia, roughness=prough, minor_loss=0)

G = wn.get_graph()

list(G.edges(data=True))
G.get_edge_data('0BEC76A8', '0BEC76BA')
G.get_edge_data('0BEC76BA', '0BEC98AA')
G.get_edge_data('2FB8DE67','2FB8DE68')

lengths={}
for edge in G.edges():
   startnode=edge[0]
   endnode=edge[1]
   lengths[edge]=round(math.sqrt(((pos_dict0[endnode][1]-pos_dict0[startnode][1])**2)+((pos_dict0[endnode][0]-pos_dict0[startnode][0])**2)),2)
lengths['0BEC76BA', '0BEC98AA']
lengths['0BEC76A8', '0BEC76BA']
lengths['2FB8DE67','2FB8DE68']
type(lengths)
calc_length_df = DataFrame.from_dict(lengths, orient = 'index')
calc_length_df.head()
#####################################################################################
###make a nested dictionary for the above? That would give us the pipe designation###
#####################################################################################
nx.draw_networkx(G, with_labels = False, node_size = 5, pos = pos_dict0, arrows = False)

length_dict = defaultdict(list)
for i, j, k in zip(resultspipes_nodupes_df.PipeLength,resultspipes_nodupes_df.FacilityToNodeName,resultspipes_nodupes_df.FacilityFromNodeName):
    length_dict[i].append(j)
    length_dict[i].append(k)
length_dict0 = dict(length_dict)

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size
print("Current size:", fig_size)

wntr.graphics.plot_network(wn)

#wn.write_inpfile(filename = '/Users/aya/Documents/code-pfs/gas-nx/small-networkzerotwo.inp', units = 'CFS')
