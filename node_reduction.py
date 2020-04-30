import pandas as pd
from pandas import DataFrame as df

#toplevelpath = "/Users/kavyaub/Documents/mySubjects/ConEdison"
toplevelpath = "/Users/aya/Documents/code-pfs/gas-nx"
all_datadir = "/NYU_LeakData"

def ask_user_path(pathinput, datadirinput):
    ask_user_path_text = 'FilePath for Data is: ' + pathinput + datadirinput + ' OK? y / n '
    response = 'y'
    user_inputYN = input(ask_user_path_text)
    if user_inputYN.lower() not in response:
        new_input = 'PASTE FULL PATH TO YOUR DATA DIRECTORY HERE: '
        newpath = input(new_input)
        return newpath
    elif user_inputYN in response:
        response2 = pathinput + datadirinput
        return response2

datadirpath = ask_user_path(toplevelpath, all_datadir)
#get_file_datadirpath = ask_user_path(toplevelpath, all_datadir)
print(datadirpath)

def get_file(name):
    anomaly = datadirpath+name
    nFile=pd.read_csv(anomaly)
    return nFile

def reducer(input_df, template_df):
    unique_nodes = template_df.NAME.unique()
    reduced_nodeArr = input_df[input_df.NAME.isin(unique_nodes)]
    reduced_nodeArr.reset_index(inplace = True, drop = True)
    return reduced_nodeArr

preDir = "/ReducedNodeSet/"
name1k="CECnodes_1k_TableToExcel"
name2k="CECnodes_2k_TableToExcel"
name3k="CECnodes_3k_TableToExcel"
name500="CECnodes_500_TableToExcel"
filetype = ".csv"
nodes1k = get_file(preDir+name1k+filetype)
nodes2k = get_file(preDir+name2k+filetype)
nodes3k = get_file(preDir+name3k+filetype)
nodes500 = get_file(preDir+name500+filetype)

preDir = "/LeakData_ZeroDegrees/"
leakFree48 = "NYU Anamoly Data_ZeroDeg_Nodes.csv"
name0_01="NYU Anamoly Data_ZeroDeg_Nodes_Leak1.csv"
name0_11="NYU Anamoly Data_ZeroDeg_Nodes_Leak11.csv"
name0_21="NYU Anamoly Data_ZeroDeg_Nodes_Leak21.csv"
name0_31="NYU Anamoly Data_ZeroDeg_Nodes_Leak31.csv"
name0_41="NYU Anamoly Data_ZeroDeg_Nodes_Leak41.csv"
leak0_00 = get_file(preDir+leakFree48)
leak0_01 = get_file(preDir+name0_01)
leak0_11 = get_file(preDir+name0_11)
leak0_21 = get_file(preDir+name0_21)
leak0_31 = get_file(preDir+name0_31)
leak0_41 = get_file(preDir+name0_41)

leak0_11_1k = reducer(leak0_11, nodes1k)

print(leak0_11_1k.shape)
