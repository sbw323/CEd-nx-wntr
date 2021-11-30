import subprocess

from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import math
import os, shutil
import glob
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier



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
    preDir = "Data_Files/"
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
    for i in np.arange(0, len(anomaly0_500.NAME)*5, 5):
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
    f.close()

    ''' Flow AI models logic begins'''

    with open('Mymatrix.txt') as f:
        content = f.readlines()
    X = [float(x.strip()) for x in content]
    X_flow = X

    with open('Flow_anomaly_free.txt') as f:
        content = f.readlines()
    anomaly_free_pipes = [float(x.strip()) for x in content]

    with open('Flow_pipe_names.txt') as f:
        content = f.readlines()
    names = [x.strip() for x in content]

    P1, P2, P3, P4 = 10, 20, 40, 60
    P = [P4, P3, P2, P1]

    keys = ['E5', 'E4', 'E3', 'E2', 'E1', 'D5', 'D4', 'D3', 'D2', 'D1', 'C5', 'C4', 'C3', 'C2', 'C1', 'B5', 'B4', 'B3',
            'B2', 'B1', 'A5', 'A4', 'A3', 'A2', 'A1']
    colors = [5, 5, 5, 5, 4, 5, 5, 4, 4, 3, 4, 4, 3, 3, 2, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1]

    red_index, orange_index, yellow_index, green_index, blue_index= [], [], [], [], []

    for i, val in enumerate(colors):
        if val == 5:
            red_index.append(i)
        elif val == 4:
            orange_index.append(i)
        elif val == 3:
            yellow_index.append(i)
        elif val == 2:
            green_index.append(i)
        else:
            blue_index.append(i)

    highest_an_free = max(anomaly_free_pipes)
    NVD1 = P1 / highest_an_free
    NVD2 = P2 / highest_an_free
    NVD3 = P3 / highest_an_free
    NVD4 = P4 / highest_an_free

    mapping_columns = {'NAME': 'A1', 'A4D5': 'B1', 'A4D4': 'C1', 'A4D3': 'D1', 'A4D2': 'E1', 'A4D1': 'F1', 'A3D5': 'G1',
                       'A3D4': 'H1', 'A3D3': 'I1', 'A3D2': 'J1', 'A3D1': 'K1', 'A2D5': 'L1', 'A2D4': 'M1', 'A2D3': 'N1',
                       'A2D2': 'O1', 'A2D1': 'P1', 'A1D5': 'Q1', 'A1D4': 'R1', 'A1D3': 'S1', 'A1D20': 'T1',
                       'A1D1': 'U1', 'Red': 'V1', 'Orange': 'W1', 'Yellow': 'X1', 'Green': 'Y1'}
    total_nodes = len(names)
    svm_sheet, ann_sheet, actual_sheet, result_sheet, comparison_sheet = 2, 1, 3, 8, 9
    filename = 'finalLabels.xlsx'
    accuracy_metrics = [[0]*20]*20
    loss_metrics = [[0] * 20] * 20
    all_svm = [[0] * 20] * total_nodes
    all_ann = [[0] * 20] * total_nodes
    all_actual = [[0] * 20] * total_nodes
    A3 = []
    perf = []

    q, z, itr = 1, 1, 0



    for k in range(1, 5):
        A3 = []
        for l in X_flow:
            A3.append(1 if l > P[k - 1] else 0)


        indici5 = [i for i, x in enumerate(A3) if x == 1]

        D5 = [0] * len(indici5)
        for i in range(len(indici5) - 4):
            for j in range(len(indici5)):
                if indici5[i + 1] == indici5[i] + 1 and indici5[i + 2] == indici5[i] + 2 and indici5[i + 3] == indici5[i] + 3 and indici5[i + 4] == indici5[i] + 4:
                    D5[i] = indici5[i]
                    D5[i + 1] = indici5[i] + 1
                    D5[i + 2] = indici5[i] + 2
                    D5[i + 3] = indici5[i] + 3
                    D5[i + 4] = indici5[i] + 4
                    continue
                    if D5[i + j] > indici5[i] + j:
                        break
                    else:
                        D5[i] = 0
                        D5[i + 1] = 0
                        D5[i + 2] = 0
                        D5[i + 3] = 0
                        D5[i + 4] = 0
        indici = indici5


        D4 = [0] * len(indici)
        for i in range(1, len(indici) - 3):
            for j in range(1, len(indici)):
                if indici[i + 1] == indici[i] + 1 and indici[i + 2] == indici[i] + 2 and indici[i + 3] == indici[i] + 3:
                    D4[i] = indici5[i]
                    D4[i + 1] = indici5[i] + 1
                    D4[i + 2] = indici5[i] + 2
                    D4[i + 3] = indici5[i] + 3
                    continue
                    if D[i + j] > indici5[i] + j:
                        break
                    else:
                        D4[i] = 0
                        D4[i + 1] = 0
                        D4[i + 2] = 0
                        D4[i + 3] = 0
        indici2 = indici5
        D3 = [0] * len(indici2)
        for i in range(1, len(indici2) - 2):
            for j in range(1, len(indici2)):
                if indici2[i + 1] == indici2[i] + 1 and indici2[i + 2] == indici2[i] + 2:
                    D3[i] = indici5[i]
                    D3[i + 1] = indici5[i] + 1
                    D3[i + 2] = indici5[i] + 2
                    continue
                    if D[i + j] > indici5[i] + j:
                        break
                    else:
                        D3[i] = 0
                        D3[i + 1] = 0
                        D3[i + 2] = 0
        indici3 = indici5
        D2 = [0] * len(indici3)
        for i in range(1, len(indici3) - 1):
            for j in range(1, len(indici3)):
                if indici3[i + 1] == indici3[i] + 1:
                    D2[i] = indici5[i]
                    D2[i + 1] = indici5[i] + 1
                    continue
                    if D[i + j] > indici5[i] + j:
                        break
                    else:
                        D2[i] = 0
                        D2[i + 1] = 0
        indici4 = indici5
        D1 = indici5
        days = [[] * 5] * len(indici5)

        days[0].append(D5)
        days[1].append(D4)
        days[2].append(D3)
        days[3].append(D2)
        days[4].append(D1)


        for p in range(1, 6):
            D = []

            for i in days[p - 1][-1]:
                if i != 0:
                    D.append(i)
            #D = list(filter(None, days[p - 1]))

            Ysvm_new = [0] * len(X)  # X --> MyMatrix.txt

            for i in D:
                Ysvm_new[i] = 1

            Xsvm_new = X

            column_name = 'A' + str(5 - k) + 'D' + str(6 - p)
            newArr_actual = [0] * int(round(len(Ysvm_new) / 5, 1))
            j = 0
            for i in np.arange(0, len(Ysvm_new) - 5, 5):
                if Ysvm_new[i] == 1 or Ysvm_new[i + 1] == 1 or Ysvm_new[i + 2] == 1 or Ysvm_new[i + 3] == 1 or Ysvm_new[i + 4] == 1:
                    newArr_actual[j] = 1
                j += 1
            temp = []

            temp = [i for i in newArr_actual if i != 0]

            all_actual[itr] = newArr_actual

            X = loadtxt('Mymatrix.txt')
            t = Ysvm_new
            X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=42)

            clf = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=(10, 1)).fit(X_train.reshape(-1, 1), t_train)
            y_predicted = clf.predict(X_test.reshape(-1, 1))

            #e = t_test - y_predicted
            perf_val = clf.score(np.array(t_test).reshape(-1, 1), y_predicted.reshape(-1, 1))

            perf.append(perf_val)
            #e = t - perf
            loss_metrics[itr][itr] = 100 - perf_val
            itr += 1

            op = y_predicted
            yfinal = [0] * len(op)
            for i in range(len(op)):
                if op[i] <= 0.5:
                    yfinal[i] = 0
                else:
                    if op[i] > 0.5:
                        yfinal[i] = 1




    print('loss_metrics')
    print(loss_metrics)
    print('yfinal')
    print(yfinal)


    #print(len(newArr_actual))





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

