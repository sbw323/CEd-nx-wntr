%Load Data
load Mymatrix_node.txt

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

red_index=[];
orange_index=[];
yellow_index=[];
green_index=[];
blue_index=[];

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
%Sheet number to save SVM output
svm_sheet=5;
%Sheet number to save ANN output
ann_sheet=4;
%Sheet number to save Actual data
actual_sheet=6;
%Sheet number to save consolidated result
result_sheet=7;
%Sheet that contains the comparison between actual and predicted
comparison_sheet=10;

%Name of file to save the data to
filename = 'finalLabels.xlsx';

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
    days=zeros(5,length(indici5));
    days(1,:)=D5;
    days(2,:)=D4;
    days(3,:)=D3;
    days(4,:)=D2;
    days(5,:)=D1;
    
    %Following loop repeats for all the 5 scenarios
    for p=1:5
        D=nonzeros(days(p,:));

        Ysvm_new=zeros(length(X_pressure),1);
        Ysvm_new(D)=1;
        Xsvm_new = Mymatrix_node(:,1);
        
        %column_name corresponds to the name of column in Excel File
        column_name=strcat('A',num2str(5-k),'D',num2str(6-p));
        
        %Gathering all actual values
        newArr_actual=zeros(round(length(Ysvm_new)/5),1);
        j=1;
        
        %Reducing nodes to reduced_node_set number of nodes
        for i=1:5:length(Ysvm_new)-5

            if Ysvm_new(i)==1 || Ysvm_new(i+1)==1 || Ysvm_new(i+2)==1 || Ysvm_new(i+3)==1 || Ysvm_new(i+4)==1
                newArr_actual(j)=1;
            end
            j=j+1;
        end

        temp = nonzeros(newArr_actual);
        
        %Writing all actual values to Excel file

        A=table(newArr_actual,'VariableNames',...
            {column_name});
        writetable(A,filename,'Sheet',actual_sheet,'Range',mapping_columns(column_name));
        
        all_actual(itr,:)=newArr_actual;
        
        %ANN
        %When there is a popup of ANN performance matrix
        %1: You can check out the confusion matrix, ROC plot and error
        %   for each of the cells of the matrix
        %2: Click on x(close), when done exploring.
        %NOTE: Popup occurs for each cell.
        x = Mymatrix_node';

        trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

        % Create a Pattern Recognition Network
        hiddenLayerSize = 10;
        net = patternnet(hiddenLayerSize, trainFcn);

        % Setup Division of Data for Training, Validation, Testing
        net.divideParam.trainRatio = 80/100;
        net.divideParam.valRatio = 10/100;
        net.divideParam.testRatio = 10/100;

        % Train the Network
        t=Ysvm_new';
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

        
        %Reducing nodes to reduced_node_set number of nodes
        newArr_ann=zeros(round(length(yfinal)/5),1);
        j=1;
       
        for i=1:5:length(yfinal)-5
            if yfinal(i)==1 || yfinal(i+1)==1 || yfinal(i+2)==1 || yfinal(i+3)==1 || yfinal(i+4)==1
                newArr_ann(j)=1;
            end
            j=j+1;
        end
        
        %Saving value to excel
        A=table(newArr_ann,'VariableNames',...
            {column_name});
        filename = 'finalLabels.xlsx';
        writetable(A,filename,'Sheet',ann_sheet,'Range',mapping_columns(column_name));
        
        all_ann(itr,:)=newArr_ann;

        percentErrors = sum(tind ~= yind)/numel(tind);
        
        nonzero_svm=nonzeros(Ysvm_new);
        %Checking if any of the y labels is 1
        %Done because SVM cannot work with single class
        
        if nonzero_svm ~= 0

        %Train an SVM classifier using the processed data set.
            SVMModel = fitcsvm(Xsvm_new,Ysvm_new);

            classOrder = SVMModel.ClassNames;
            Order = unique(Ysvm_new)

            CVSVMModel = crossval(SVMModel);
            
            func = @(Xtrain,ytrain,Xtest,ytest)confusionmat(ytest,classify(Xtest,Xtrain,ytrain),'Order',Order)
            
            rng('default');
            cvp = cvpartition(Ysvm_new,'Kfold',10);
            
            confMat = crossval(func,Xsvm_new,Ysvm_new,'Partition',cvp);
            cvMat = reshape(sum(confMat),[],4);
            confusionchart(cvMat,order)
            
            classLoss = kfoldLoss(CVSVMModel);
            accuracy_metrics(q)=1-classLoss;
            q=q+1;

            ScoreSVMModel = fitSVMPosterior(SVMModel);

            ScoreTransform = CVSVMModel.ScoreTransform;
            W = Mymatrix_node(:,1);

            [label,score] = predict(SVMModel,W);
            Sc=nonzeros(label);
            
            %Uncomment the below to see SVM output
            %table(label,'VariableNames',...
            %    {'PredictedLabel'})
            
            
            %Reducing nodes to reduced_node_set number of nodes
            newArr_svm=zeros(round(length(label)/5),1);
            j=1;
            for i=1:5:length(label)-5
                if label(i)==1 || label(i+1)==1 || label(i+2)==1 || label(i+3)==1 || label(i+4)==1
                    newArr_svm(j)=1;
                end
                j=j+1;
            end

            A=table(newArr_svm,'VariableNames',...
                {column_name});
            filename = 'finalLabels.xlsx';
            writetable(A,filename,'Sheet',svm_sheet,'Range',mapping_columns(column_name));
            
            all_svm(itr,:)=newArr_svm;
        else
            A=table(newArr_actual,'VariableNames',...
                {column_name});
            filename = 'finalLabels.xlsx';
            writetable(A,filename,'Sheet',svm_sheet,'Range',mapping_columns(column_name));
            all_svm(itr,:)=newArr_actual;
        end
        itr=itr+1;
    end
end

%%
%Categorizing into red, orange, yellow and green
%For SVM, ANN and Actual

red_svm=zeros(total_nodes,1);
red_ann=zeros(total_nodes,1);
red_actual=zeros(total_nodes,1);

orange_svm=zeros(total_nodes,1);
orange_ann=zeros(total_nodes,1);
orange_actual=zeros(total_nodes,1);

yellow_svm=zeros(total_nodes,1);
yellow_ann=zeros(total_nodes,1);
yellow_actual=zeros(total_nodes,1);

green_svm=zeros(total_nodes,1);
green_ann=zeros(total_nodes,1);
green_actual=zeros(total_nodes,1);


for i=1:length(newArr_actual)
    for j=1:length(red_index)
        if all_svm(red_index(j),i)==1
            red_svm(i)=1;
        end
    end
    for j=1:length(orange_index)
        if red_svm(i)==0 && (all_svm(orange_index(j),i)==1)
            orange_svm(i)=1;
        end
    end
    for j=1:length(yellow_index)
        if (red_svm(i)==0 && orange_svm(i)==0) &&( all_svm(yellow_index(j),i)==1)
            yellow_svm(i)=1;
        end
    end
    for j=1:length(green_index)
        if (red_svm(i)==0 && orange_svm(i)==0 && yellow_svm(i)==0) && all_svm(green_index(j),i)==1
            green_svm(i)=1;
        end
    end
end

for i=1:length(newArr_actual)
    for j=1:length(red_index)
        if all_actual(red_index(j),i)==1
            red_actual(i)=1;
        end
    end
    for j=1:length(orange_index)
        if red_actual(i)==0 && (all_actual(orange_index(j),i)==1)
            orange_actual(i)=1;
        end
    end
    for j=1:length(yellow_index)
        if (red_actual(i)==0 && orange_actual(i)==0) &&( all_actual(yellow_index(j),i)==1)
            yellow_actual(i)=1;
        end
    end
    for j=1:length(green_index)
        if (red_actual(i)==0 && orange_actual(i)==0 && yellow_actual(i)==0) && all_actual(green_index(j),i)==1
            green_actual(i)=1;
        end
    end
end

for i=1:length(newArr_actual)
    for j=1:length(red_index)
        if all_ann(red_index(j),i)==1
            red_ann(i)=1;
        end
    end
    for j=1:length(orange_index)
        if red_ann(i)==0 && (all_ann(orange_index(j),i)==1)
            orange_ann(i)=1;
        end
    end
    for j=1:length(yellow_index)
        if (red_ann(i)==0 && orange_ann(i)==0) &&( all_ann(yellow_index(j),i)==1)
            yellow_ann(i)=1;
        end
    end
    for j=1:length(green_index)
        if (red_ann(i)==0 && orange_ann(i)==0 && yellow_ann(i)==0) && all_ann(green_index(j),i)==1
            green_ann(i)=1;
        end
    end
end

%%
%Writing to excel

column_name='NAME';
names_table=table(names{:,1},'VariableNames',...
                {column_name});
writetable(names_table,filename,'Sheet',actual_sheet,'Range','A1');
writetable(names_table,filename,'Sheet',svm_sheet,'Range','A1');
writetable(names_table,filename,'Sheet',ann_sheet,'Range','A1');
writetable(names_table,filename,'Sheet',comparison_sheet,'Range','A1');


column_name='Red';
A=table(red_svm,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',svm_sheet,'Range',mapping_columns(column_name));
A=table(red_ann,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',ann_sheet,'Range',mapping_columns(column_name));
A=table(red_actual,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',actual_sheet,'Range',mapping_columns(column_name));

column_name='Orange';
A=table(orange_svm,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',svm_sheet,'Range',mapping_columns(column_name));
A=table(orange_ann,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',ann_sheet,'Range',mapping_columns(column_name));
A=table(orange_actual,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',actual_sheet,'Range',mapping_columns(column_name));


column_name='Yellow';
A=table(yellow_svm,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',svm_sheet,'Range',mapping_columns(column_name));
A=table(yellow_ann,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',ann_sheet,'Range',mapping_columns(column_name));
A=table(yellow_actual,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',actual_sheet,'Range',mapping_columns(column_name));


column_name='Green';
A=table(green_svm,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',svm_sheet,'Range',mapping_columns(column_name));
A=table(green_ann,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',ann_sheet,'Range',mapping_columns(column_name));
A=table(green_actual,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',actual_sheet,'Range',mapping_columns(column_name));


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
writetable(final_table,filename,'Sheet',7,'Range','B10','WriteRowNames',true);

%%

%Computing the SVM accuracy to excel computed using ClassLoss
red_metric_loss=[accuracy_metrics(1),accuracy_metrics(2),accuracy_metrics(3),accuracy_metrics(4),accuracy_metrics(6),accuracy_metrics(7)];
if isnan(sum(red_metric_loss)/length(nonzeros(red_metric_loss)))==1;red_metric_loss=100; else; red_metric_loss=sum(red_metric_loss)/length(nonzeros(red_metric_loss))*100; end
orange_metric_loss=[accuracy_metrics(5),accuracy_metrics(8),accuracy_metrics(9),accuracy_metrics(11),accuracy_metrics(12)];
if isnan(sum(orange_metric_loss)/length(nonzeros(orange_metric_loss)))==1; orange_metric_loss=100;else; orange_metric_loss=sum(orange_metric_loss)/length(nonzeros(orange_metric_loss))*100 ; end
yellow_metric_loss=[accuracy_metrics(13),accuracy_metrics(14),accuracy_metrics(16),accuracy_metrics(17),accuracy_metrics(10)];
if isnan(sum(yellow_metric_loss)/length(nonzeros(yellow_metric_loss)))==1; yellow_metric_loss= 100;else; yellow_metric_loss=sum(yellow_metric_loss)/length(nonzeros(yellow_metric_loss))*100; end
green_metric_loss=[accuracy_metrics(15),accuracy_metrics(18),accuracy_metrics(19)];
if isnan(sum(green_metric_loss)/length(nonzeros(green_metric_loss)))==1; green_metric_loss= 100;else; green_metric_loss= sum(green_metric_loss)/length(nonzeros(green_metric_loss))*100; end


%Computing the ANN accuracy to excel computed using Performance of ANN
red_loss=[loss_metrics(1),loss_metrics(2),loss_metrics(3),loss_metrics(4),loss_metrics(6),loss_metrics(7)];
red_loss=sum(red_loss)/length(nonzeros(red_loss));
orange_loss=[loss_metrics(5),loss_metrics(8),loss_metrics(9),loss_metrics(11),loss_metrics(12)];
orange_loss=sum(orange_loss)/length(nonzeros(orange_loss));
yellow_loss=[loss_metrics(13),loss_metrics(14),loss_metrics(16),loss_metrics(17),loss_metrics(10)];
yellow_loss=sum(yellow_loss)/length(nonzeros(yellow_loss));
green_loss=[loss_metrics(15),loss_metrics(18),loss_metrics(19)];
green_loss=sum(green_loss)/length(nonzeros(green_loss));


loss_acc=[[red_metric_loss,orange_metric_loss,yellow_metric_loss,green_metric_loss];[red_loss,orange_loss,yellow_loss,green_loss]];

%Writing SVM and ANN accuracies to result sheet
final_table=array2table(loss_acc,'RowNames',headers,'VariableNames',colors);
writetable(final_table,filename,'Sheet',7,'Range','B14','WriteRowNames',true);


%%

%Output the comparison matrix to excel
%Place Actual next to Predicted
%Also printing the total percentage of the different colors
red_actual_percentage=red_actual;
red_actual_percentage(total_nodes)=(length(nonzeros(red_actual))/total_nodes)*100;
column_name='RedActual';
A=table(red_actual_percentage,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',comparison_sheet,'Range','B1');

red_svm_percentage=red_svm;
red_svm_percentage(total_nodes)=(length(nonzeros(red_svm))/total_nodes)*100;
column_name='RedSVMPredicted';
A=table(red_svm_percentage,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',comparison_sheet,'Range','C1');

red_ann_percentage=red_ann;
red_ann_percentage(total_nodes)=(length(nonzeros(red_ann))/total_nodes)*100;
column_name='RedANNPredicted';
A=table(red_ann_percentage,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',comparison_sheet,'Range','D1');


orange_actual_percentage=orange_actual;
orange_actual_percentage(total_nodes)=(length(nonzeros(orange_actual))/total_nodes)*100;
column_name='OrangeActual';
A=table(orange_actual_percentage,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',comparison_sheet,'Range','E1');

orange_svm_percentage=orange_svm;
orange_svm_percentage(total_nodes)=(length(nonzeros(orange_svm))/total_nodes)*100;
column_name='OrangeSVMPredicted';
A=table(orange_svm_percentage,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',comparison_sheet,'Range','F1');

orange_ann_percentage=orange_ann;
orange_ann_percentage(total_nodes)=(length(nonzeros(orange_ann))/total_nodes)*100;
column_name='OrangeANNPredicted';
A=table(orange_ann_percentage,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',comparison_sheet,'Range','G1');


yellow_actual_percentage=yellow_actual;
yellow_actual_percentage(total_nodes)=(length(nonzeros(yellow_actual))/total_nodes)*100;
column_name='YellowActual';
A=table(yellow_actual_percentage,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',comparison_sheet,'Range','H1');

yellow_svm_percentage=yellow_svm;
yellow_svm_percentage(total_nodes)=(length(nonzeros(yellow_svm))/total_nodes)*100;
column_name='YellowSVMPredicted';
A=table(yellow_svm_percentage,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',comparison_sheet,'Range','I1');

yellow_ann_percentage=yellow_ann;
yellow_ann_percentage(total_nodes)=(length(nonzeros(yellow_ann))/total_nodes)*100;
column_name='YellowANNPredicted';
A=table(yellow_ann_percentage,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',comparison_sheet,'Range','J1');


green_actual_percentage=green_actual;
green_actual_percentage(total_nodes)=(length(nonzeros(green_actual))/total_nodes)*100;
column_name='GreenActual';
A=table(green_actual_percentage,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',comparison_sheet,'Range','K1');

green_svm_percentage=green_svm;
green_svm_percentage(total_nodes)=(length(nonzeros(green_svm))/total_nodes)*100;
column_name='GreenSVMPredicted';
A=table(green_svm_percentage,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',comparison_sheet,'Range','L1');

green_ann_percentage=green_ann;
green_ann_percentage(total_nodes)=(length(nonzeros(green_ann))/total_nodes)*100;
column_name='GreenANNPredicted';
A=table(green_ann_percentage,'VariableNames',...
                {column_name});
writetable(A,filename,'Sheet',comparison_sheet,'Range','M1');

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


