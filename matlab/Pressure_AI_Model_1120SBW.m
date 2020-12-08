%Load Data
load Mymatrix_node.txt
load Mymatrix_32node.txt

% load Training Data
X = abs(Mymatrix_node(:,1));
X_pressure= X(:,1);

%load Anomaly Free Data
an_free=fopen('Pressure_anomaly_free.txt','r');
anomaly_free=textscan(an_free,'%s');
fclose(an_free);
anomaly_free_nodes=zeros(length(anomaly_free{1,:}),1);
for i=1:length(anomaly_free{1,:})
    anomaly_free_nodes(i)=str2double(cell2mat(anomaly_free{1}(i)));
end

%%
%Loading names of reducerd node set generated from FlowDataAnalysis.txt
names_id=fopen('Pressure_node_names.txt','r');
names=textscan(names_id,'%s');
fclose(names_id);

%%
%Input the pressure deviation amplitude
%Change default values for threshold as requried
%Input your selected values below

%Threshold 1
P1=0.3;
%Threshold 2
P2=0.5;
%Threshold 3
P3=0.8;
%Threshold 4
P4=1.1;

P=[P4,P3,P2,P1];

%%
%Consider the first row(> Threshold A4) of the matrix as E, 
%frequency of 1 is depicted by E1, frequency 2 is E2 ...frequency 5 is E5
%Row 2 (>Threshold A3) is depicted by D, Row 3 is C, Row 4 is B 
%and Row 5 is A
keys = {'E5','E4','E3','E2','E1','D5','D4','D3','D2','D1','C5','C4','C3','C2','C1','B5','B4','B3','B2','B1','A5','A4','A3','A2','A1'};
%Colors - 5 refers to red, 4 refers to orange,
% 3 refers to yellow, 2 refers to green
% 1 refers to blue
colors = [5,5,5,5,4,5,5,4,4,3,4,4,3,3,2,3,3,2,2,1,1,1,1,1,1];

colors_code=containers.Map(keys,colors);
for i=1:length(colors)
    disp(keys(i)+":"+colors(i));
end


red_index=find(colors==5);
orange_index=find(colors==4);
yellow_index=find(colors==3);
green_index=find(colors==2);
blue_index=find(colors==1);



%Highest velocity in anomaly free and P1 - 10, NVD , this number/max
%anomaly fre
%TODO:

highest_an_free=mean(anomaly_free_nodes);
NPD1=P1/highest_an_free;
NPD2=P2/highest_an_free;
NPD3=P3/highest_an_free;
NPD4=P4/highest_an_free;

%Pressure deviation - divide by the pressure in anomaly free, not the max
% NVD1, NVD2, NVD3 and NVD4
%%

%Define this as per the number of nodes returned
%from reducer function in PressureDataAnalysis.m
total_nodes=length(names{1,:});
%Sheet number to save SVMmax output
svm_max_sheet=1;
%Sheet number to save ANNmax output
ann_max_sheet=2;
%Sheet number to save SVM output
svm_sheet=3;
%Sheet number to save ANN output
ann_sheet=4;
%Sheet number to save Actual data
actual_sheet=5;
%Sheet number to save consolidated result
result_sheet=6;
%Sheet that contains the comparison between actual and predicted
comparison_sheet=7;

%Name of file to save the data to
%filename = 'finalLabels.xlsx';
filename = 'finalLabelsmax_new.xlsx';

%Total number of entries
total_entries=5;


%Map of columns to excel column name
keySet = {'A4D5','A4D4','A4D3','A4D2','A4D1','A3D5','A3D4','A3D3','A3D2','A3D1','A2D5','A2D4','A2D3','A2D2','A2D1','A1D5','A1D4','A1D3','A1D2','A1D1','Red','Orange','Yellow','Green'};
valueSet = {'B1','C1', 'D1', 'E1', 'F1', 'G1' ,'H1', 'I1', 'J1', 'K1', 'L1', 'M1', 'N1', 'O1', 'P1', 'Q1', 'R1', 'S1' ,'T1','U1','V1','W1','X1','Y1'};
mapping_columns=containers.Map(keySet,valueSet);

%%
%This matrix tracks the SVM accuracy computed using classLoss
accuracy_metrics=zeros(20);
q=1;

%This matrix tracks the ANN accuracy computed using confusion matrix
loss_metrics=zeros(20);
z=1;

all_svm=zeros(20,total_nodes);
all_svm_max=zeros(20,total_nodes);

all_ann=zeros(20,total_nodes);
all_actual=zeros(20,total_nodes);
itr=1;
%%

%The following loop runs 4 times for each threshold.
for k=1:4
    for l=1:length(X_pressure)
        %P(k) replaces the threshold
        A3= X_pressure>P(k);
    end
    A3=A3';
    indici5=find(A3==1);
    %Computing all the indices for all the 5 scenarios
    %that repeat over 5 scenarios

% Uncomment this section for 5

     D5=zeros(1,length(indici5));
      for i=1:length(indici5)-4
      for j=1:length(indici5)
         if indici5(i)==indici5(i)
         if indici5(i+1)==indici5(i)+1
         if indici5(i+2)==indici5(i)+2
         if indici5(i+3)==indici5(i)+3
         if indici5(i+4)==indici5(i)+4
         D5(i)=indici5(i);
         D5(i+1)=indici5(i)+1;
         D5(i+2)=indici5(i)+2;
         D5(i+3)=indici5(i)+3;
         D5(i+4)=indici5(i)+4;
         continue;
         if D5(i+j)>indici5(i)+j
         break;
         else
         D5(i)=0;
         D5(i+1)=0;
         D5(i+2)=0;
         D5(i+3)=0; 
         D5(i+4)=0;
         end
         end
         end
         end
         end
         end
      end
      end

% Uncomment this section for 4 

     indici = indici5;
     %Computing all the indices that repeat for 4 scenarios
      
     D4=zeros(1,length(indici));
      for i=1:length(indici)-3
      for j=1:length(indici)
         if indici(i)==indici(i)
         if indici(i+1)==indici(i)+1
         if indici(i+2)==indici(i)+2
         if indici(i+3)==indici(i)+3
         D4(i)=indici5(i);
         D4(i+1)=indici5(i)+1;
         D4(i+2)=indici5(i)+2;
         D4(i+3)=indici5(i)+3;
         continue;
         if D(i+j)>indici5(i)+j
         break;
         else
         D4(i)=0;
         D4(i+1)=0;
         D4(i+2)=0;
         D4(i+3)=0;  
         end
         end
         end
         end
         end
      end
      end


    indici2=indici5;
     %Computing all the indices that repeat for 3 scenarios   
     D3=zeros(1,length(indici2));
     for i=1:length(indici2)-2
     for j=1:length(indici2)
         if indici2(i)==indici2(i)
         if indici2(i+1)==indici2(i)+1
         if indici2(i+2)==indici2(i)+2
         D3(i)=indici5(i);
         D3(i+1)=indici5(i)+1;
         D3(i+2)=indici5(i)+2;
         continue;
         if D(i+j)>indici5(i)+j
         break;
         else
         D3(i)=0;
         D3(i+1)=0;
         D3(i+2)=0; 
         end
         end
         end
         end
    end
    end


    indici3=indici5;
     %Computing all the indices that repeat for 2 scenarios
     D2=zeros(1,length(indici3));
     for i=1:length(indici3)-1
     for j=1:length(indici3)
         if indici3(i)==indici3(i)
         if indici3(i+1)==indici3(i)+1
         D2(i)=indici5(i);
         D2(i+1)=indici5(i)+1;
         continue;
         if D(i+j)>indici5(i)+j
         break;
         else
         D2(i)=0;
         D2(i+1)=0;
         end
         end
         end
     end
     end

    indici4=indici5;
    %Computing all the indices that repeat for at least 1 scenario
    D1=indici5;
    days=zeros(total_entries,length(indici5));
 %Section for 5
     days(1,:)=D5;
     days(2,:)=D4;
     days(3,:)=D3;
     days(4,:)=D2;
     days(5,:)=D1;

%Section for 3
%    days(1,:)=D3;
%    days(2,:)=D2;
%    days(3,:)=D1;

    %Computing all the indices for all the 5 scenarios
    %that repeat over 5 scenarios

    %Following loop repeats for all the 5 scenarios
    for p=1:total_entries
        D=nonzeros(days(p,:));

        Ysvm_new=zeros(length(X_pressure),1);
        Ysvm_new(D)=1;
        Xsvm_new = Mymatrix_node(:,1);
        
        %column_name corresponds to the name of column in Excel File
        column_name=strcat('A',num2str(5-k),'D',num2str(total_entries+1-p));
        
        %Reducing nodes to reduced_node_set number of nodes 
        newArr_actual=compute_res(Ysvm_new,total_entries);

        temp = nonzeros(newArr_actual);
        
        %Writing all actual values to Excel file
        writeToExcel(newArr_actual,column_name,filename,actual_sheet,mapping_columns);
        
        all_actual(itr,:)=newArr_actual;
        
        %ANN
        %When there is a popup of ANN performance matrix
        %1: You can check out the confusion matrix, ROC plot and error
        %   for each of the cells of the matrix
        %2: Click on x(close), when done exploring.
        %NOTE: Popup occurs for each cell.
        x = Mymatrix_node;
        x2 = Mymatrix_32node;

        trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

        % Create a Pattern Recognition Network
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize, trainFcn);

        % Setup Division of Data for Training, Validation, Testing
        net.divideParam.trainRatio = 80/100;
        net.divideParam.valRatio = 10/100;
        net.divideParam.testRatio = 10/100;

        % Train the Network
        t=Ysvm_new;
        [net,tr] = train(net,x,t);

        % Test the Network
        y = net(x);
        e = gsubtract(t,y);
        performance = perform(net,t,y);
        %Computing Accuracy using ANN metrics
        loss_metrics(z)=100.0-performance;
        z=z+1;
        tind = vec2ind(t);
        yind = vec2ind(y);

        op=y;
        
        %Manually converting all probablities above 0.5 to 1
        %and all other probablities to 0
        %This is done so because ANN gives us the probablities,
        %but we need 0 or 1

        yfinal=zeros(1,length(op));
        for i=1:length(op)
            if op(i)<=0.5
                yfinal(i)=0;
            else
                if op(i)>0.5
                    yfinal(i)=1;
                end
            end
        end
        
        y2 = net(x2);
        e = gsubtract(t,y2);
        performance = perform(net,t,y2);
        %Computing Accuracy using ANN metrics
        loss_metrics(z)=100.0-performance;
        z=z+1;
        tind = vec2ind(t);
        y2ind = vec2ind(y2);

        op2=y2;
        
        %Manually converting all probablities above 0.5 to 1
        %and all other probablities to 0
        %This is done so because ANN gives us the probablities,
        %but we need 0 or 1

        y2final=zeros(1,length(op2));
        for i=1:length(op2)
            if op2(i)<=0.5
                y2final(i)=0;
            else
                if op2(i)>0.5
                    y2final(i)=1;
                end
            end
        end

        
        %Reducing nodes to reduced_node_set number of nodes
        newArr_ann=compute_res(yfinal,total_entries);
        newArr_ann2=compute_res(y2final,total_entries);

        
        %Saving value to excel
        writeToExcel(newArr_ann2,column_name,filename,ann_max_sheet,mapping_columns);
        
        all_ann(itr,:)=newArr_ann2;

        percentErrors = sum(tind ~= y2ind)/numel(tind);
        
        nonzero_svm=nonzeros(Ysvm_new);

        %Checking if any of the y labels is 1
        %Done because SVM cannot work with single class
        
        if nonzero_svm ~= 0

        %Train an SVM classifier using the processed data set.
            SVMModel = fitcsvm(Xsvm_new,Ysvm_new);

            classOrder = SVMModel.ClassNames;

            CVSVMModel = crossval(SVMModel);
            classLoss = kfoldLoss(CVSVMModel);
            accuracy_metrics(q)=1-classLoss;
            q=q+1;

            ScoreSVMModel = fitSVMPosterior(SVMModel);

            ScoreTransform = CVSVMModel.ScoreTransform;
            W = Mymatrix_node(:,1);
            Z = Mymatrix_32node(:,1);

            [label,score] = predict(SVMModel,W);
            [label_max,score_max] = predict(SVMModel,Z);

            Sc=nonzeros(label);
            Scc=nonzeros(label_max);

            
            %Uncomment the below to see SVM output
            %table(label,'VariableNames',...
            %    {'PredictedLabel'})
            
            
            %Reducing nodes to reduced_node_set number of nodes
            newArr_svm=compute_res(label,total_entries);            
            newArrmax_svm=compute_res(label_max,total_entries);
            
            writeToExcel(newArrmax_svm,column_name,filename,svm_max_sheet,mapping_columns);

            writeToExcel(newArr_svm,column_name,filename,svm_sheet,mapping_columns);
            
            all_svm(itr,:)=newArr_svm;
            all_svm_max(itr,:)=newArrmax_svm;

        else
            writeToExcel(newArr_actual,column_name,filename,svm_sheet,mapping_columns);
            all_svm(itr,:)=newArr_actual;
            
            writeToExcel(newArr_actual,column_name,filename,svm_max_sheet,mapping_columns);
            all_svm_max(itr,:)=newArr_actual;

        end
        itr=itr+1;
    end
end

%%
%Categorizing into red, orange, yellow and green
%For SVM, ANN and Actual

[red_svm,orange_svm,yellow_svm,green_svm]=segregate_colors(total_nodes, colors, all_svm,newArr_actual);
[red_svmmax,orange_svmmax,yellow_svmmax,green_svmmax]=segregate_colors(total_nodes, colors, all_svm_max,newArr_actual);
[red_ann,orange_ann,yellow_ann,green_ann]=segregate_colors(total_nodes, colors, all_ann,newArr_actual);
[red_actual,orange_actual,yellow_actual,green_actual]=segregate_colors(total_nodes, colors, all_actual,newArr_actual);


%%
%Writing to excel

column_name='NAME';
names_table=table(names{:,1},'VariableNames',...
                {column_name});
writetable(names_table,filename,'Sheet',actual_sheet,'Range','A1');
writetable(names_table,filename,'Sheet',svm_sheet,'Range','A1');
writetable(names_table,filename,'Sheet',ann_sheet,'Range','A1');
writetable(names_table,filename,'Sheet',comparison_sheet,'Range','A1');
writetable(names_table,filename,'Sheet',svm_max_sheet,'Range','A1');



column_name='Red';
writeToExcel(red_svm,column_name,filename,svm_sheet,mapping_columns);
writeToExcel(red_svmmax,column_name,filename,svm_sheet,mapping_columns);
writeToExcel(red_ann,column_name,filename,ann_sheet,mapping_columns);
writeToExcel(red_actual,column_name,filename,actual_sheet,mapping_columns);


column_name='Orange';
writeToExcel(orange_svm,column_name,filename,svm_sheet,mapping_columns);
writeToExcel(orange_svmmax,column_name,filename,svm_sheet,mapping_columns);
writeToExcel(orange_ann,column_name,filename,ann_sheet,mapping_columns);
writeToExcel(orange_actual,column_name,filename,actual_sheet,mapping_columns);


column_name='Yellow';
writeToExcel(yellow_svm,column_name,filename,svm_sheet,mapping_columns);
writeToExcel(yellow_svmmax,column_name,filename,svm_sheet,mapping_columns);
writeToExcel(yellow_ann,column_name,filename,ann_sheet,mapping_columns);
writeToExcel(yellow_actual,column_name,filename,actual_sheet,mapping_columns);


column_name='Green';
writeToExcel(green_svm,column_name,filename,svm_sheet,mapping_columns);
writeToExcel(green_svmmax,column_name,filename,svm_sheet,mapping_columns);
writeToExcel(green_ann,column_name,filename,ann_sheet,mapping_columns);
writeToExcel(green_actual,column_name,filename,actual_sheet,mapping_columns);


%%
%Computing the total number of reds, oranges, yellows, greens and blues
headers={'Actual','ANN','SVM'};
colors={'Red','Orange','Yellow','Green','Blue'};
total_blues_actual=total_nodes-(length(nonzeros(red_actual))+length(nonzeros(orange_actual))+length(nonzeros(yellow_actual))+length(nonzeros(green_actual)));
total_blues_svm=total_nodes-(length(nonzeros(red_svm))+length(nonzeros(orange_svm))+length(nonzeros(yellow_svm))+length(nonzeros(green_svm)));
total_blues_ann=total_nodes-(length(nonzeros(red_ann))+length(nonzeros(orange_ann))+length(nonzeros(yellow_ann))+length(nonzeros(green_ann)));

%%
%Overall Accuracy
total_reds=[length(nonzeros(red_actual)),length(nonzeros(red_ann)),length(nonzeros(red_svm))];
total_oranges=[length(nonzeros(orange_actual)),length(nonzeros(orange_ann)),length(nonzeros(orange_svm))];
total_yellows=[length(nonzeros(yellow_actual)),length(nonzeros(yellow_ann)),length(nonzeros(yellow_svm))];
total_greens=[length(nonzeros(green_actual)),length(nonzeros(green_ann)),length(nonzeros(green_svm))];
total_blues=[total_blues_actual,total_blues_ann,total_blues_svm];

total_red_actual=length(nonzeros(red_actual));
total_orange_actual=length(nonzeros(orange_actual));
total_yellow_actual=length(nonzeros(yellow_actual));
total_green_actual=length(nonzeros(green_actual));

percentage_red=total_red_actual/total_nodes;
percentage_orange=total_orange_actual/total_nodes;
percentage_yellow=total_yellow_actual/total_nodes;
percentage_green=total_green_actual/total_nodes;
percentage_blue=total_blues_actual/total_nodes;


disp("percentage reds:"+(percentage_red));
disp("percentage oranges:"+(percentage_orange));
disp("percentage yellows:"+(percentage_yellow));
disp("percentage green:"+(percentage_green));
disp("percentage blues:"+(percentage_blue));
%%
finals=[total_reds;total_oranges;total_yellows;total_greens;total_blues];

final_table=array2table(finals,'RowNames',colors,'VariableNames',headers);

writetable(final_table,filename,'Sheet',7,'Range','B2','WriteRowNames',true);

%%
%Computing the accuracy for reds, oranges, yellows and greens
%Based on predictions from ANN and SVM matching the actual values

headers={'SVM','ANN'};

%Saving matching accuracy to Excel
redsvm_acc=calculate_acc(red_actual,red_svm,total_nodes);
redann_acc=calculate_acc(red_actual,red_ann,total_nodes);
orangesvm_acc=calculate_acc(orange_actual,orange_svm,total_nodes);
orangeann_acc=calculate_acc(orange_actual,orange_ann,total_nodes);
yellowsvm_acc=calculate_acc(yellow_actual,yellow_svm,total_nodes);
yellowann_acc=calculate_acc(yellow_actual,yellow_ann,total_nodes);
greensvm_acc=calculate_acc(green_actual,green_svm,total_nodes);
greenann_acc=calculate_acc(green_actual,green_ann,total_nodes);
matching=[[redsvm_acc,orangesvm_acc,yellowsvm_acc,greensvm_acc];[redann_acc,orangeann_acc,yellowann_acc,greenann_acc]];

%Writing the accuracy to result_sheet of excel
colors={'Red','Orange','Yellow','Green'};
final_table=array2table(matching,'RowNames',headers,'VariableNames',colors);
writetable(final_table,filename,'Sheet',result_sheet,'Range','B10','WriteRowNames',true);

%%

%Computing the SVM accuracy to excel computed using ClassLoss

red_metric_loss=compute_avg(red_index,accuracy_metrics);


orange_metric_loss=compute_avg(orange_index,accuracy_metrics);

yellow_metric_loss=compute_avg(yellow_index,accuracy_metrics);

green_metric_loss=compute_avg(green_index,accuracy_metrics);


if isnan(red_metric_loss)==1;red_metric_loss=100; else; red_metric_loss=red_metric_loss*100; end
if isnan(orange_metric_loss)==1;orange_metric_loss=100; else; orange_metric_loss=orange_metric_loss*100; end
if isnan(yellow_metric_loss)==1;yellow_metric_loss=100; else; yellow_metric_loss=yellow_metric_loss*100; end
if isnan(green_metric_loss)==1;green_metric_loss=100; else; green_metric_loss=green_metric_loss*100; end


%Computing the ANN accuracy to excel computed using Performance of ANN

red_loss=compute_avg(red_index,loss_metrics);
% red_loss=[loss_metrics(1),loss_metrics(2),loss_metrics(3),loss_metrics(4),loss_metrics(6),loss_metrics(7)];
% red_loss=sum(red_loss)/length(nonzeros(red_loss));

orange_loss=compute_avg(orange_index,loss_metrics);
% orange_loss=[loss_metrics(5),loss_metrics(8),loss_metrics(9),loss_metrics(11),loss_metrics(12)];
% orange_loss=sum(orange_loss)/length(nonzeros(orange_loss));
% yellow_loss=[loss_metrics(13),loss_metrics(14),loss_metrics(16),loss_metrics(17),loss_metrics(10)];
% yellow_loss=sum(yellow_loss)/length(nonzeros(yellow_loss));
yellow_loss=compute_avg(yellow_index,loss_metrics);

% green_loss=[loss_metrics(15),loss_metrics(18),loss_metrics(19)];
% green_loss=sum(green_loss)/length(nonzeros(green_loss));

green_loss=compute_avg(green_index,loss_metrics);

loss_acc=[[red_metric_loss,orange_metric_loss,yellow_metric_loss,green_metric_loss];[red_loss,orange_loss,yellow_loss,green_loss]];

%Writing SVM and ANN accuracies to result sheet
final_table=array2table(loss_acc,'RowNames',headers,'VariableNames',colors);
writetable(final_table,filename,'Sheet',result_sheet,'Range','B14','WriteRowNames',true);


%%

%Output the comparison matrix to excel
%Place Actual next to Predicted
%Also printing the total percentage of the different colors
column_name='RedActual';
excelFormatter( filename,red_actual,total_nodes,comparison_sheet, column_name,'B1');

column_name='RedSVMPredicted';
excelFormatter( filename,red_svm,total_nodes,comparison_sheet, column_name,'C1');

column_name='RedANNPredicted';
excelFormatter( filename,red_ann,total_nodes,comparison_sheet, column_name,'D1');

column_name='OrangeActual';
excelFormatter( filename,orange_actual,total_nodes,comparison_sheet, column_name,'E1');


column_name='OrangeSVMPredicted';
excelFormatter( filename,orange_svm,total_nodes,comparison_sheet, column_name,'F1');

column_name='OrangeANNPredicted';
excelFormatter( filename,orange_ann,total_nodes,comparison_sheet, column_name,'G1');


column_name='YellowActual';
excelFormatter( filename,yellow_actual,total_nodes,comparison_sheet, column_name,'H1');

column_name='YellowSVMPredicted';
excelFormatter( filename,yellow_svm,total_nodes,comparison_sheet, column_name,'I1');

column_name='YellowANNPredicted';
excelFormatter( filename,yellow_ann,total_nodes,comparison_sheet, column_name,'J1');

column_name='GreenActual';
excelFormatter( filename,green_actual,total_nodes,comparison_sheet, column_name,'K1');

column_name='GreenSVMPredicted';
excelFormatter( filename,green_svm,total_nodes,comparison_sheet, column_name,'L1');

column_name='GreenANNPredicted';
excelFormatter( filename,green_ann,total_nodes,comparison_sheet, column_name,'M1');

%%

%Computes the accuracy based on the number of matches
function acc=calculate_acc(actual,other,total_nodes)
    ctr=0;
    for i=1:length(actual)
        if actual(i)==other(i)
            ctr=ctr+1;
        end
    end
    
    acc=(ctr/total_nodes)*100;
end

function res=compute_res(input_arr,total_entries)
    jj=1;
    res=zeros(round(length(input_arr)/total_entries),1);
    for i=1:total_entries:length(input_arr)-total_entries
        for ent=i:i+total_entries
            if input_arr(ent)==1
                res(jj)=1;
                break;
             end
         end
         jj=jj+1;
    end
end


function writeToExcel(arr,column_name,filename,sheet_name,mapping_columns)
    A=table(arr,'VariableNames',...
                {column_name});
    disp(column_name);
    writetable(A,filename,'Sheet',sheet_name,'Range',mapping_columns(column_name));
end

        
function [red,orange,yellow,green]=segregate_colors(total_nodes, colors, arr,newArr_actual)
    red=zeros(total_nodes,1);
    orange=zeros(total_nodes,1);
    yellow=zeros(total_nodes,1);
    green=zeros(total_nodes,1);
    
    red_index=find(colors==5);
    orange_index=find(colors==4);
    yellow_index=find(colors==3);
    green_index=find(colors==2);
    
    for i=1:length(newArr_actual)
        for j=1:length(red_index)
            if arr(red_index(j),i)==1
                red(i)=1;
            end
        end
        for j=1:length(orange_index)
            if red(i)==0 && (arr(orange_index(j),i)==1)
                orange(i)=1;
            end
        end
        for j=1:length(yellow_index)
            if (red(i)==0 && orange(i)==0) &&( arr(yellow_index(j),i)==1)
                yellow(i)=1;
            end
        end
        for j=1:length(green_index)
            if (red(i)==0 && orange(i)==0 && yellow(i)==0) && arr(green_index(j),i)==1
                green(i)=1;
            end
        end
    end
end

function res=compute_avg(index,arr)
    res=0;
    for j=1:length(index)
        res=res+arr(index(j));
    end
    res=res/length(nonzeros(res));
end

function excelFormatter( filename,actual,total_nodes,sheet, column_name,cell_num)
    red_actual_percentage=actual;
    red_actual_percentage(total_nodes)=(length(nonzeros(actual))/total_nodes)*100;
    A=table(red_actual_percentage,'VariableNames',...
                {column_name});
    writetable(A,filename,'Sheet',sheet,'Range',cell_num);
end