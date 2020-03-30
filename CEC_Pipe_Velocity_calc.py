
#######################
## Importing Modules ##
#######################

import pandas as pd
from pandas import DataFrame as df
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import math

######################################
## Your personal filepath goes here ##
## Just toggle to MD w/hashes after ##
######################################

#KB_toplevelpath = 0
#KB_datadir = 0

SBW_toplevelpath = "/Users/aya/Documents/code-pfs/gas-nx"
SBW_datadir = "/NYU_LeakData"

def ask_user_path(toplevelpath, datadir):
    ask_user_path_text = 'FilePath for Data is: ' + toplevelpath + datadir + ' OK? y / n'
    response = 'y'
    user_inputYN = input(ask_user_path_text)
    if user_inputYN.lower() not in response:
        new_input = 'PASTE FULL PATH TO YOUR DATA DIRECTORY HERE: '
        newpath = input(new_input)
        return newpath
    elif user_inputYN in response:
        response2 = toplevelpath + datadir
        return response2

datadirpath = ask_user_path(SBW_toplevelpath, SBW_datadir)

###########################
## Populating DataFrames ##
###########################

def get_file(name):
    anomaly = datadirpath+name
    nFile=pd.read_csv(anomaly)
    return nFile

preDir = "/LeakData_ZeroDegrees/"
leaklist = ["NYU Anamoly Data_ZeroDeg_Pipes_Leak11.csv","NYU Anamoly Data_ZeroDeg_Pipes_Leak21.csv", "NYU Anamoly Data_ZeroDeg_Pipes_Leak31.csv", "NYU Anamoly Data_ZeroDeg_Pipes_Leak41.csv"]

leak0_11 = get_file(preDir+leaklist[0])
leak0_21 = get_file(preDir+leaklist[1])
leak0_31 = get_file(preDir+leaklist[2])
leak0_41 = get_file(preDir+leaklist[3])

##################################
## Testing Velocity Calculation ##
##################################

pid = leak0_41['NAME'].iloc[0]
pffa = leak0_41['FacilityFlowAbsolute'].iloc[0]
pdia = leak0_41['PipeDiameter'].iloc[0]
print(pid, "has a diameter in inches of:", pdia, "and a flow rate of:", pffa, "in mcfph")

parea = pdia**2/4*math.pi*in2_ft2
print(pid, "has an area of:", parea, "in ft^2")

pvelo = pffa/parea*1000/3600
print(pid, "has a velocity of:", pvelo, "in fps")

###################################################
### Calculating Pipe Velocity Rates in an Array ###
###################################################

setOfNames = set(leak0_41['NAME'])
if leak0_41['PipeDiameter'].iloc[0] == leak0_41.iloc[0][94]:
    print("pipe diameter is row 94")
if leak0_41['FacilityFlowAbsolute'].iloc[0] == leak0_41.iloc[0][90]:
    print("pipe flow is row 90")
in2_ft2 = 0.0005787

## Leak 41 ##
#Defintion creates an array of the node information with an appended Zeros matrix#

def pipe_velocity_calc(pipedf):
    final_pipedf = np.array([pipedf['NAME'],pipedf['FacilityFromNodeName'],pipedf['FacilityToNodeName'],pipedf['FacilityFlowAbsolute'],pipedf['PipeDiameter'],[0]*pipedf.NAME.size,[0]*pipedf.NAME.size,[0]*pipedf.NAME.size,[0]*pipedf.NAME.size,[0]*pipedf.NAME.size,[0]*pipedf.NAME.size])
    setOfNames = set(pipedf['NAME'])

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

final_leak0_11 = pipe_velocity_calc(leak0_11)
final_leak0_21 = pipe_velocity_calc(leak0_21)
final_leak0_31 = pipe_velocity_calc(leak0_31)
final_leak0_41 = pipe_velocity_calc(leak0_41)

if final_leak0_41['VELOpipeFPS'].iloc[0] == pvelo:
    print("validated")
veloarray11 = np.array(final_leak0_11['VELOpipeFPS'])
veloarray21 = np.array(final_leak0_21['VELOpipeFPS'])
veloarray31 = np.array(final_leak0_31['VELOpipeFPS'])
veloarray41 = np.array(final_leak0_41['VELOpipeFPS'])
print(np.mean(veloarray11),np.std(veloarray11))
print(np.mean(veloarray21),np.std(veloarray21))
print(np.mean(veloarray31),np.std(veloarray31))
print(np.mean(veloarray41),np.std(veloarray41))

print(362*10**6*0.014/10**6*0.5)
