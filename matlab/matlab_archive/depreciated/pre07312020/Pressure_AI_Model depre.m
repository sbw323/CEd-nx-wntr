%Load Data
load Mymatrix_node.txt %training


% load Training Data
X = abs(Mymatrix_node(:,1)); 
X_pressure= X(:,1);


% % Threshold 1
% 

%P0=0.3;
P1=0.3;

% 
% % Threshold 2
P2=0.5;
% 
% % Threshold 3
P3=0.8;
% 
% % Threshold 4

P4=1.1;

P=[P4,P3,P2,P1];

%Map of columns to excel column name
keySet = {'A4D5','A4D4','A4D3','A4D2','A4D1','A3D5','A3D4','A3D3','A3D2','A3D1','A2D5','A2D4','A2D3','A2D2','A2D1','A1D5','A1D4','A1D3','A1D2','A1D1','Red','Orange','Yellow','Green'};
valueSet = {'B1','C1', 'D1', 'E1', 'F1', 'G1' ,'H1', 'I1', 'J1', 'K1', 'L1', 'M1', 'N1', 'O1', 'P1', 'Q1', 'R1', 'S1' ,'T1','U1','V1','W1','X1','Y1'};
mapping_columns=containers.Map(keySet,valueSet);
accuracy_metrics=zeros(20);
q=1;

loss_metrics=zeros(20);
z=1;

total_nodes=204;
svm_sheet=5;
ann_sheet=4;
actual_sheet=6;
result_sheet=7;
filename = 'finalLabels.xlsx';


all_svm=zeros(20,total_nodes);
all_ann=zeros(20,total_nodes);
all_actual=zeros(20,total_nodes);
itr=1;


% Severe anomaly
for k=1:4
    for l=1:length(X_pressure)
        A3= X_pressure>P(k);
    end
    A3=A3';
    indici5=find(A3==1);

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
    D1=indici5;
    days=zeros(5,length(indici5));
    days(1,:)=D5;
    days(2,:)=D4;
    days(3,:)=D3;
    days(4,:)=D2;
    days(5,:)=D1;
    
    for p=1:5
        D=nonzeros(days(p,:));

        Ysvm_new=zeros(length(X_pressure),1);
        Ysvm_new(D)=1;
        Xsvm_new = Mymatrix_node(:,1);
        
        column_name=strcat('A',num2str(5-k),'D',num2str(6-p));

        newArr_actual=zeros(round(length(Ysvm_new)/5),1);
        j=1;
        for i=1:5:length(Ysvm_new)-5

            if Ysvm_new(i)==1 || Ysvm_new(i+1)==1 || Ysvm_new(i+2)==1 || Ysvm_new(i+3)==1 || Ysvm_new(i+4)==1
                newArr_actual(j)=1;
            end
            j=j+1;
        end

        temp = nonzeros(newArr_actual);
        
        A=table(newArr_actual,'VariableNames',...
            {column_name});
        writetable(A,filename,'Sheet',actual_sheet,'Range',mapping_columns(column_name));
        
        all_actual(itr,:)=newArr_actual;
        
        %ANN
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
        loss_metrics(z)=100.0-performance;
        z=z+1;
        tind = vec2ind(t);
        yind = vec2ind(y);

        op=y;

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


        newArr_ann=zeros(round(length(yfinal)/5),1);
        j=1;
        for i=1:5:length(yfinal)-5
            if yfinal(i)==1 || yfinal(i+1)==1 || yfinal(i+2)==1 || yfinal(i+3)==1 || yfinal(i+4)==1
                newArr_ann(j)=1;
            end
            j=j+1;
        end
        
        
        A=table(newArr_ann,'VariableNames',...
            {column_name});
        filename = 'finalLabels.xlsx';
        writetable(A,filename,'Sheet',ann_sheet,'Range',mapping_columns(column_name));
        
        all_ann(itr,:)=newArr_ann;

        percentErrors = sum(tind ~= yind)/numel(tind);
        
        nonzero_svm=nonzeros(Ysvm_new);
        
        if nonzero_svm ~= 0

        % Train an SVM classifier using the processed data set.
            SVMModel = fitcsvm(Xsvm_new,Ysvm_new);

            classOrder = SVMModel.ClassNames;

            CVSVMModel = crossval(SVMModel);
            classLoss = kfoldLoss(CVSVMModel);
            accuracy_metrics(q)=1-classLoss;
            q=q+1;

            ScoreSVMModel = fitSVMPosterior(SVMModel);

            ScoreTransform = CVSVMModel.ScoreTransform;
            W = Mymatrix_node(:,1);

            [label,score] = predict(SVMModel,W);
            Sc=nonzeros(label);

            disp(Ysvm_new);
            table(label,'VariableNames',...
                {'PredictedLabel'})

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
    if all_svm(1,i)==1 || all_svm(2,i)==1 || all_svm(3,i)==1 || all_svm(4,i)==1||all_svm(6,i)==1|| all_svm(7,i)==1
        red_svm(i)=1;
    end
    if red_svm(i)==0 &&( all_svm(5,i)==1 || all_svm(8,i)==1 || all_svm(9,i)==1||all_svm(11,i)==1|| all_svm(12,i)==1)
        orange_svm(i)=1;
    end
    if (red_svm(i)==0 && orange_svm(i)==0) &&( all_svm(13,i)==1 || all_svm(14,i)==1 || all_svm(16,i)==1||all_svm(17,i)==1|| all_svm(10,i)==1)
        yellow_svm(i)=1;
    end
    if (red_svm(i)==0 && orange_svm(i)==0 && yellow_svm(i)==0) &&( all_svm(15,i)==1 || all_svm(18,i)==1 || all_svm(19,i)==1)
        green_svm(i)=1;
    end
end

for i=1:length(newArr_actual)
    if all_actual(1,i)==1 || all_actual(2,i)==1 || all_actual(3,i)==1 || all_actual(4,i)==1||all_actual(6,i)==1|| all_actual(7,i)==1
        red_actual(i)=1;
    end
    if red_actual(i)==0 &&( all_actual(5,i)==1 || all_actual(8,i)==1 || all_actual(9,i)==1||all_actual(11,i)==1|| all_actual(12,i)==1)
        orange_actual(i)=1;
    end
    if (red_actual(i)==0 && orange_actual(i)==0) &&( all_actual(13,i)==1 || all_actual(14,i)==1 || all_actual(16,i)==1||all_actual(17,i)==1|| all_actual(10,i)==1)
        yellow_actual(i)=1;
    end
    if (red_actual(i)==0 && orange_actual(i)==0 && yellow_actual(i)==0) &&( all_actual(15,i)==1 || all_actual(18,i)==1 || all_actual(19,i)==1)
        green_actual(i)=1;
    end
end

for i=1:length(newArr_actual)
    if all_ann(1,i)==1 || all_ann(2,i)==1 || all_ann(3,i)==1 || all_ann(4,i)==1||all_ann(6,i)==1|| all_ann(7,i)==1
        red_ann(i)=1;
    end
    if red_ann(i)==0 &&( all_ann(5,i)==1 || all_ann(8,i)==1 || all_ann(9,i)==1||all_ann(11,i)==1|| all_ann(12,i)==1)
        orange_ann(i)=1;
    end
    if (red_ann(i)==0 && orange_ann(i)==0) &&( all_ann(13,i)==1 || all_ann(14,i)==1 || all_ann(16,i)==1||all_ann(17,i)==1|| all_ann(10,i)==1)
        yellow_ann(i)=1;
    end
    if (red_ann(i)==0 && orange_ann(i)==0 && yellow_ann(i)==0) &&( all_ann(15,i)==1 || all_ann(18,i)==1 || all_ann(19,i)==1)
        green_ann(i)=1;
    end
end

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



headers={'Actual','ANN','SVM'};
colors={'Red','Orange','Yellow','Green','Blue'};
total_blues_actual=204-(length(nonzeros(red_actual))+length(nonzeros(orange_actual))+length(nonzeros(yellow_actual))+length(nonzeros(green_actual)));
total_blues_svm=204-(length(nonzeros(red_svm))+length(nonzeros(orange_svm))+length(nonzeros(yellow_svm))+length(nonzeros(green_svm)));
total_blues_ann=204-(length(nonzeros(red_ann))+length(nonzeros(orange_ann))+length(nonzeros(yellow_ann))+length(nonzeros(green_ann)));

total_reds=[length(nonzeros(red_actual)),length(nonzeros(red_ann)),length(nonzeros(red_svm))];
total_oranges=[length(nonzeros(orange_actual)),length(nonzeros(orange_ann)),length(nonzeros(orange_svm))];
total_yellows=[length(nonzeros(yellow_actual)),length(nonzeros(yellow_ann)),length(nonzeros(yellow_svm))];
total_greens=[length(nonzeros(green_actual)),length(nonzeros(green_ann)),length(nonzeros(green_svm))];
total_blues=[total_blues_actual,total_blues_ann,total_blues_svm];

finals=[total_reds;total_oranges;total_yellows;total_greens;total_blues];

final_table=array2table(finals,'RowNames',colors,'VariableNames',headers);

writetable(final_table,filename,'Sheet',7,'Range','B2','WriteRowNames',true);


headers={'SVM','ANN'};


redsvm_acc=calculate_acc(red_actual,red_svm);
redann_acc=calculate_acc(red_actual,red_ann);
orangesvm_acc=calculate_acc(orange_actual,orange_svm);
orangeann_acc=calculate_acc(orange_actual,orange_ann);
yellowsvm_acc=calculate_acc(yellow_actual,yellow_svm);
yellowann_acc=calculate_acc(yellow_actual,yellow_ann);
greensvm_acc=calculate_acc(green_actual,green_svm);
greenann_acc=calculate_acc(green_actual,green_ann);
matching=[[redsvm_acc,orangesvm_acc,yellowsvm_acc,greensvm_acc];[redann_acc,orangeann_acc,yellowann_acc,greenann_acc]];

colors={'Red','Orange','Yellow','Green'};
final_table=array2table(matching,'RowNames',headers,'VariableNames',colors);
writetable(final_table,filename,'Sheet',7,'Range','B10','WriteRowNames',true);


red_metric_loss=[accuracy_metrics(1),accuracy_metrics(2),accuracy_metrics(3),accuracy_metrics(4),accuracy_metrics(6),accuracy_metrics(7)];
if isnan(sum(red_metric_loss)/length(nonzeros(red_metric_loss)))==1;red_metric_loss=100; else; red_metric_loss=sum(red_metric_loss)/length(nonzeros(red_metric_loss))*100; end
orange_metric_loss=[accuracy_metrics(5),accuracy_metrics(8),accuracy_metrics(9),accuracy_metrics(11),accuracy_metrics(12)];
if isnan(sum(orange_metric_loss)/length(nonzeros(orange_metric_loss)))==1; orange_metric_loss=100;else; orange_metric_loss=sum(orange_metric_loss)/length(nonzeros(orange_metric_loss))*100 ; end
yellow_metric_loss=[accuracy_metrics(13),accuracy_metrics(14),accuracy_metrics(16),accuracy_metrics(17),accuracy_metrics(10)];
if isnan(sum(yellow_metric_loss)/length(nonzeros(yellow_metric_loss)))==1; yellow_metric_loss= 100;else; yellow_metric_loss=sum(yellow_metric_loss)/length(nonzeros(yellow_metric_loss))*100; end
green_metric_loss=[accuracy_metrics(15),accuracy_metrics(18),accuracy_metrics(19)];
if isnan(sum(green_metric_loss)/length(nonzeros(green_metric_loss)))==1; green_metric_loss= 100;else; green_metric_loss= sum(green_metric_loss)/length(nonzeros(green_metric_loss))*100; end



red_loss=[loss_metrics(1),loss_metrics(2),loss_metrics(3),loss_metrics(4),loss_metrics(6),loss_metrics(7)];
red_loss=sum(red_loss)/length(nonzeros(red_loss));
orange_loss=[loss_metrics(5),loss_metrics(8),loss_metrics(9),loss_metrics(11),loss_metrics(12)];
orange_loss=sum(orange_loss)/length(nonzeros(orange_loss));
yellow_loss=[loss_metrics(13),loss_metrics(14),loss_metrics(16),loss_metrics(17),loss_metrics(10)];
yellow_loss=sum(yellow_loss)/length(nonzeros(yellow_loss));
green_loss=[loss_metrics(15),loss_metrics(18),loss_metrics(19)];
green_loss=sum(green_loss)/length(nonzeros(green_loss));


loss_acc=[[red_metric_loss,orange_metric_loss,yellow_metric_loss,green_metric_loss];[red_loss,orange_loss,yellow_loss,green_loss]];


final_table=array2table(loss_acc,'RowNames',headers,'VariableNames',colors);
writetable(final_table,filename,'Sheet',7,'Range','B14','WriteRowNames',true);





function acc=calculate_acc(actual,other)
    ctr=0;
    for i=1:length(actual)
        if actual(i)==other(i)
            ctr=ctr+1;
        end
    end
    
    acc=ctr/204;
end


