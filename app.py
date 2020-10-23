import subprocess

from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import math
import os, shutil
import glob



import sqlite3 as sql
from pathlib import Path

in2_ft2 = 0.0005787
setOfNames = []
nodes200 = ''

app = Flask(__name__)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
else:
    files = glob.glob('/uploads/*')
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    fetchFiles()
    return render_template("index.html")

@app.route('/', methods=['POST'])
def upload_file():

    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        flash('File(s) successfully uploaded')
        fetchFiles()
        return redirect(url_for('index'))

def fetchFiles():

    print('inside fetch Files')
    preDir = "/Data_Files/"
    name_0 = "NYU Anamoly Data_ZeroDeg_Pipes"
    name_1 = "NYU Anamoly Data_ZeroDeg_Pipes_Leak1"
    name0_11 = "NYU Anamoly Data_ZeroDeg_Pipes_Leak11"
    name0_21 = "NYU Anamoly Data_ZeroDeg_Pipes_Leak21"
    name0_31 = "NYU Anamoly Data_ZeroDeg_Pipes_Leak31"
    name0_41 = "NYU Anamoly Data_ZeroDeg_Pipes_Leak41"
    name200 = "CECnodes_200_TableToExcel"

    filetype = ".csv"

    nFile0 = get_file(preDir + name_0 + filetype)
    nFile1 = get_file(preDir + name_1 + filetype)
    leak0_11 = get_file(preDir + name0_11 + filetype)
    leak0_21 = get_file(preDir + name0_21 + filetype)
    leak0_31 = get_file(preDir + name0_31 + filetype)
    leak0_41 = get_file(preDir + name0_41 + filetype)
    global nodes200
    nodes200 = get_file(preDir + name200 + filetype)

    if nFile0 is None or nFile1 is None or leak0_11 is None or leak0_21 is None or leak0_31 is None or leak0_41 is None or nodes200 is None:
        flash('Few required files are missing')

        return

    global setOfNames
    setOfNames = set(leak0_41['NAME'])

    an_free = pipe_velocity_calc(nFile0)
    anomaly4 = pipe_velocity_calc(leak0_41)
    anomaly3 = pipe_velocity_calc(leak0_31)
    anomaly2 = pipe_velocity_calc(leak0_21)
    anomaly1 = pipe_velocity_calc(leak0_11)
    anomaly0 = pipe_velocity_calc(nFile1)
    anomaly0_500 = reducer(anomaly0)
    anomaly1_500 = reducer(anomaly1)
    anomaly2_500 = reducer(anomaly2)
    anomaly3_500 = reducer(anomaly3)
    anomaly4_500 = reducer(anomaly4)
    an_free_500 = reducer(an_free)
    temps = np.vstack(
        [anomaly0_500.VELOpipeFPS, anomaly1_500.VELOpipeFPS, anomaly2_500.VELOpipeFPS, anomaly3_500.VELOpipeFPS,
         anomaly4_500.VELOpipeFPS])

    an_free_1 = np.hstack(
        [an_free_500.VELOpipeFPS, an_free_500.VELOpipeFPS, an_free_500.VELOpipeFPS, an_free_500.VELOpipeFPS,
         an_free_500.VELOpipeFPS])
    temps = temps.T

    new_arr = temps


    for i in range(0, len(temps)):
        mins = min(temps[i])
        if an_free_500.VELOpipeFPS[i] < mins:
            mins = an_free_500.VELOpipeFPS[i]
            new_arr[i] -= an_free_500.VELOpipeFPS[i]
        else:
            if temps[i][0] == mins:
                new_arr[i] -= temps[i][0]

            elif temps[i][1] == mins:
                new_arr[i][0] = 0
                new_arr[i][1] = 0
                new_arr[i][2] -= temps[i][1]
                new_arr[i][3] -= temps[i][1]
                new_arr[i][4] -= temps[i][1]

            elif temps[i][2] == mins:
                new_arr[i][0] = 0
                new_arr[i][1] = 0
                new_arr[i][2] = 0
                new_arr[i][3] -= temps[i][2]
                new_arr[i][4] -= temps[i][2]
            elif temps[i][3] == mins:
                new_arr[i][0] = 0
                new_arr[i][1] = 0
                new_arr[i][2] = 0
                new_arr[i][3] = 0
                new_arr[i][4] -= temps[i][3]
            else:
                new_arr[i] = 0

    #np.reshape(new_arr, 1025)
    j = 0

    saved = np.zeros((len(anomaly0_500.NAME)*5, 1))
    for i in np.arange(0, len(anomaly0_500.NAME), 5):
        saved[i] = new_arr[j][0]
        saved[i + 1] = new_arr[j][1]
        saved[i + 2] = new_arr[j][2]
        saved[i + 3] = new_arr[j][3]
        saved[i + 4] = new_arr[j][4]
        j = j + 1

    f = open("Flow_anomaly_free.txt", "wt")

    for i in range(0, len(an_free_500)):
        f.write(str(an_free_500.VELOpipeFPS[i]))
        f.write('\n')
    f.close()

    f = open('Mymatrix.txt', 'wt')
    for i in range(0, len(saved)):
        f.write(str(saved[i][0]))
        f.write('\n')
    f.close()

    f = open('Flow_pipe_names.txt', 'wt');
    for i in range(0, len(anomaly1_500)):
        f.write(str(anomaly1_500.NAME[i]))
        f.write('\n')
    f.close()

    # open and read the file after the appending:
    f = open("Flow_anomaly_free.txt", "r")
    print(f.read())
    return

def get_file(name):
    #get_file_datadirpath = 'uploads/'
    #anomaly = get_file_datadirpath+name
    anomaly = name
    #my_file = Path(anomaly)

    nfile = pd.read_csv(anomaly)

    return nfile


def reducer(input_arr):

    global nodes200
    unique_nodes = nodes200.NAME.unique()
    reduced_nodeArr = input_arr[input_arr.FacilityFromNodeName.isin(unique_nodes)]
    reduced_nodeArr1 = input_arr[input_arr.FacilityToNodeName.isin(unique_nodes)]

    reduced_nodeArr.reset_index(inplace=True, drop=True)
    reduced_nodeArr1.reset_index(inplace=True, drop=True)

    df = pd.merge(reduced_nodeArr, reduced_nodeArr1)
    df_new = df.drop_duplicates()
    return df_new


def flowDeviation_vel(file0, file1):
    res_arr = file1
    res_arr['VELOpipeFPS']= file1.VELOpipeFPS
    dividend = max(file0['VELOpipeFPS'])
    res_arr.VELOpipeFPS = abs(res_arr.VELOpipeFPS.subtract(file0.VELOpipeFPS))/dividend
    return res_arr


def calculations(input_arr):
    final_temp_arr = np.array(
        [input_arr['NAME'], input_arr['FacilityFromNodeName'], input_arr['FacilityToNodeName'], input_arr['NAME'],
         [0] * input_arr.NAME.size, [0] * input_arr.NAME.size, [0] * input_arr.NAME.size, [0] * input_arr.NAME.size,
         [0] * input_arr.NAME.size, [0] * input_arr.NAME.size])

    anomalyFreeNode = "Data_Files/LeakData_ZeroDegrees/NYU Anamoly Data_ZeroDeg_Nodes.csv"
    nodeArr = pd.read_csv(anomalyFreeNode)
    setOfNames = set(nodeArr['NAME'])

    for i in range(0, final_temp_arr[1].size):
        if final_temp_arr[1][i] in setOfNames:
            temp = nodeArr.loc[nodeArr['NAME'] == final_temp_arr[1][i]]
            final_temp_arr[4][i] = temp.iloc[0][3]
            final_temp_arr[5][i] = temp.iloc[0][2]
        if final_temp_arr[2][i] in setOfNames:
            temp = nodeArr.loc[nodeArr['NAME'] == final_temp_arr[2][i]]
            final_temp_arr[6][i] = temp.iloc[0][3]
            final_temp_arr[7][i] = temp.iloc[0][2]

    mid_elem_x = (final_temp_arr[4] + final_temp_arr[6]) / 2.0
    mid_elem_y = (final_temp_arr[5] + final_temp_arr[7]) / 2.0

    final_temp_arr[8] = mid_elem_x
    final_temp_arr[9] = mid_elem_y
    input_arr['mid_point_x'] = mid_elem_x
    input_arr['mid_point_y'] = mid_elem_y

def pipe_velocity_calc(pipedf):

    final_pipedf = np.array(
        [pipedf['NAME'], pipedf['FacilityFromNodeName'], pipedf['FacilityToNodeName'], pipedf['FacilityFlowAbsolute'],
         pipedf['PipeDiameter'], [0] * pipedf.NAME.size, [0] * pipedf.NAME.size, [0] * pipedf.NAME.size,
         [0] * pipedf.NAME.size, [0] * pipedf.NAME.size, [0] * pipedf.NAME.size])

    for i in range(0, final_pipedf[1].size):
        if final_pipedf[0][i] in setOfNames:
            temp = pipedf.loc[pipedf['NAME'] == final_pipedf[0][i]]
            final_pipedf[5][i] = temp.iloc[0][90]
            final_pipedf[6][i] = temp.iloc[0][94]

    ## Creating the Velocity Rate Column ##
    global in2_ft2
    elem_AREApipeFT2 = (final_pipedf[6] ** 2 / 4 * math.pi * in2_ft2)
    elem_VELOpipeFPS = (final_pipedf[5] / elem_AREApipeFT2 * 1000 / 3600)
    final_pipedf[8] = elem_AREApipeFT2
    final_pipedf[9] = elem_VELOpipeFPS
    pipedf['AREApipeFT2'] = elem_AREApipeFT2
    pipedf['VELOpipeFPS'] = elem_VELOpipeFPS

    return pipedf

if __name__ == '__main__':
    app.run(debug=True)
