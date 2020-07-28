%Load Data
load anomaly_free_pipes.txt %AnomalyFree
load latestflow_5.txt %training
load combined_validation.txt %validation

% load Anomaly free data
an_free=anomaly_free_pipes(:,:); 

% load Training Data
X = abs(latestflow_5(:,1)); 
X_flow= X(:,1);

% Set Threshold 
flow=abs(latestflow_5(:,1)); %Threshold flow all rows, column 2

flowD=abs(an_free(:,1)); %Threshold flow all rows, column 2

% Mean & STD of Anomaly Free data
flowavg=mean(flowD); %Avg of Threshold for flow = Mean of  thresholdflowD

flowSTD=std(X_flow);

% % Threshold 1
% 
flowRangeP1=flowD+0.01*flowavg;
flowRangeN1=flowD-0.01*flowavg;
flowP1=flowD+5;
flowN1=flowD-5;
P1=10;
% 
% % Threshold 2
% 
flowRangeP2=flowD+0.025*flowavg;
flowRangeN2=flowD-0.025*flowavg;
flowP2=flowD+10;
flowN2=flowD-10;
P2=20;
% 
% % Threshold 3
% 
flowRangeP3=flowD+0.055*flowavg;
flowRangeN3=flowD-0.055*flowavg;
flowP3=flowD+20;
P3=40;
% 
% % Threshold 4
% 
flowRangeP4=flowD+0.075*flowavg;
flowRangeN4=flowD-0.075*flowavg;
flowP4=flowD+40;
flowN4=flowD-40;
P4=60;

%
% D5 duration index > 5 day
% D4 duration index > 4 day
% D3 duration index > 3 day
% D2 duration index > 2 day
% D1 duration index > 1 day
% Severe anomaly
for i=1:length(X_flow)
    %A3= (X_duration>DurationRangeP3)&(X_flow>flowRangeP3);%&(thresholdHl>HLRangeP3)&(thresholdflow<flowRangeP3);
    A3= X_flow>P4;
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
% end

%  indici=zeros(1,length(D5));
%  for i=1:length(D5)
%      if D5(i)==0
%         indici(i)=indici5(i);
%      else
%      indici(i)=0;
%      end
%  end

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
% end

%  indici2=zeros(1,length(D4));
% %  for i=1:length(D4)
% %      if D4(i)==0
% %         indici2(i)=indici(i);
% %      else
% %      indici2(i)=0;
% %      end
% %  end
 
%  for i=1:length(D4)
%      if indici5(i)==D5(i)
%         indici2(i)=0;
%      else
%      if D4(i)==0
%          indici2(i)=indici5(i);
%      else
%      indici2(i)=0;
%      end
%      end
% end

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
% end
% 
%  indici3=zeros(1,length(D3));
% %  for i=1:length(D3)
% %      if indici(i)==D4(i)
% %         indici3(i)=0;
% %      else
% %      if D3(i)==0
% %          indici3(i)=indici(i);
% %      else
% %      indici3(i)=0;
% %      end
% %      end
% %  end


% for i=1:length(D3)
%      if indici5(i)==D5(i)
%         indici3(i)=0;
%      else
%          if indici5(i)==D4(i)
%             indici3(i)=0;
%          else 
%              if indici5(i)==D3(i)
%                 indici3(i)=0;
%              else
%                  if  D5(i)==0
%                    indici3(i)=indici5(i);
%                  else
%                      if D5(i)==0
%                      indici3(i)=indici5(i);
%                      else
%                      indici3(i)=0;
%                      end
%                  end
%              end
%          end
%      end
%  end
 
 
% As proceed from D4 to D1 all loops must be placed in comment because of
% the magnitude greater the Academic Matlab use

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


%  indici4=zeros(1,length(D2));
% 
%  for i=1:length(D3)
%      if indici5(i)==D5(i)
%         indici4(i)=0;
%      else
%          if indici5(i)==D4(i)
%             indici4(i)=0;
%          else 
%              if indici5(i)==D3(i)
%                 indici4(i)=0;
%              else 
%                 if indici5(i)==D2(i)
%                     indici4(i)=0;
%                 else
%                     if  D5(i)==0
%                         indici4(i)=indici5(i);
%                     else
%                         if D5(i)==0
%                             indici4(i)=indici5(i);
%                         else
%                             indici4(i)=0;
%                         end
%                     end
%                 end
%              end
%          end
%      end
%  end

indici4=indici5;
D1=indici5;

% % input SVM
% indici4= durata un'ora
D=nonzeros(D2);

%D4=ones(length(indici),1);
Xs=X(D,:);
Xs2=latestflow_5(:,1);
Xsvm=[Xs; Xs2];
% Xsvm=Xsvm(:,1);
Y11=zeros(length(latestflow_5),1);
Y22=ones(length(Xs),1);
Ysvm=[Y22; Y11];

Ysvm_new=zeros(length(latestflow_5),1);
Ysvm_new(D)=1;
Xsvm_new = latestflow_5(:,1);

% % % input SVM
% % Xs=X(D4,1:2);
% % Xs2=an_free_flow(:,1:2);
% Xs=X(D4,1);
% Xs2=an_free_flowp(:,1:2);
% Xsvm=[Xs; Xs2];
% % Xsvm=Xsvm(:,1:2);
% Xsvm=Xsvm(:,1);
% Y11=zeros(length(an_free_flowp),1);
% Y22=ones(length(Xs),1);
% Ysvm=[Y11; Y22];

% % input ANN
% indiciD1=nonzeros(D4)';
% A3indici=[1:length(A3)];
% Y1= Ysvm;
% Y2=Y1==0; 
% YY=[Y1';Y2'];
% XX=Xsvm';

% Train an SVM classifier using the processed data set.
SVMModel = fitcsvm(Xsvm_new,Ysvm_new);

% properties of |SVMModel|, for example, to determine the class order, by using
% dot notation.
classOrder = SVMModel.ClassNames;


% Plot a scatter diagram of the data and circle the support vectors.
sv = SVMModel.SupportVectors;
% figure
% gscatter(X(:,1),X(:,2),,Y)
% hold on
% plot(sv,'ko','MarkerSize',10)
% legend('no leak','leak','Support Vector')
% hold off

CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel);

ScoreSVMModel = fitSVMPosterior(SVMModel);

ScoreTransform = CVSVMModel.ScoreTransform;
W = latestflow_5(:,1);
%W = abs(combined_flows(:,1:2));
%W=[Xs2;W];
%dim=length(Xsvm);
%W=W(1:dim,:);

[label,score] = predict(SVMModel,W);
Sc=nonzeros(label);
table(label,'VariableNames',...
    {'PredictedLabel'})



