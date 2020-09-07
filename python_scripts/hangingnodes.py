import networkx as nx
import pandas as pd
from pandas import DataFrame as df
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

pnodes = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Nodes_Leak41.csv"
ppipes = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Pipes_Leak41.csv"
pipesdf0 = pd.read_csv(ppipes)
nodesdf0 = pd.read_csv(pnodes)
pipesdf0.dropna(axis = 1, how = 'all', inplace = True)
nodesdf0.dropna(axis = 1, how = 'all', inplace = True)

#for i, name in enumerate(pipesdf0.columns):
#    print(i+1, name)
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

hangingnodes
