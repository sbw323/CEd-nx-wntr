##############
#refer to https://www.idtools.com.au/3d-network-graphs-python-mplot3d-toolkit/
##############

import numpy as np
import networkx as nx
import pandas as pd
from pandas import DataFrame as df
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pnodes = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Nodes.csv"
ppipes = "/Users/aya/Documents/code-pfs/gas-nx/NYU_LeakData/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Pipes.csv"
pipesdf0 = pd.read_csv(ppipes)
nodesdf0 = pd.read_csv(pnodes)
pipesdf0.dropna(axis = 1, how = 'all', inplace = True)
nodesdf0.dropna(axis = 1, how = 'all', inplace = True)
cntrlnd = '0BEC50B8'

G = nx.Graph()

pos_dict = defaultdict(list)
for i, j, k in zip(nodesdf0.NAME,nodesdf0.NodeXCoordinate,nodesdf0.NodeYCoordinate):
    pos_dict[i].append(j)
    pos_dict[i].append(k)
pos_dict0 = dict(pos_dict)

nodepressure_dict0 = {val:item for val, item in zip(nodesdf0.NAME,nodesdf0.NodePressure)}

d3pos_dict = defaultdict(list)
for d in (pos_dict0, nodepressure_dict0): # you can list as many input dicts as you want here
    for key, value in d.items():
        d3pos_dict[key].append(value)
d3pos_dict0 = dict(d3pos_dict)

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

def network_plot_3D(G, angle, save=False):
    colors = ["#1a1aff", "#00cc00", "#ffff00", "#ffa500", "#ff4d4d"]

    # 3D network plot
    with plt.style.context(('ggplot')):

        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)

        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in d3pos_dict0.items():
            xi = value[0][0]
            yi = value[0][1]
            zi = value[1]

            # Scatter plot
            ax.scatter(xi, yi, zi, edgecolors='k', alpha=0.7)

        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i,j in enumerate(G.edges()):
            x = np.array((d3pos_dict0[j[0]][0][0], d3pos_dict0[j[1]][0][0]))
            y = np.array((d3pos_dict0[j[0]][0][1], d3pos_dict0[j[1]][0][1]))
            z = np.array((d3pos_dict0[j[0]][1], d3pos_dict0[j[1]][1]))

        # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)

    # Set the initial view
    ax.view_init(30, angle)
    # Hide the axes
    ax.set_axis_off()
    if save is not False:
        plt.savefig("/Users/aya/Documents/code-pfs/gas-nx/plots"+str(angle).zfill(3)+".png")
        plt.close('all')
    else:
         plt.show()

    return

network_plot_3D(G, 60)
