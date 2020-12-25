%Pressure data pre processing File
%Click on "Import Data" to import the Pressure files for the 5 scenarios
%Name of the files to be uploaded
%Eg: NYUAnamolyDataZeroDegNodesLeak41 refers to file with leak value of 41
degree41_nodes=NYUAnamolyData32DegNodesLeak41;
degree31_nodes=NYUAnamolyData32DegNodesLeak31;
degree21_nodes=NYUAnamolyData32DegNodesLeak21;
degree11_nodes=NYUAnamolyData32DegNodesLeak11;
degree1_nodes=NYUAnamolyData32DegNodesLeak1;
anomaly_free_nodes=NYUAnamolyData32DegNodes;
%Name of file with 200 nodes
nodes_200=CECnodes200TableToExcel;
%%
%Note to user
%For reducing to 500 nodes
%"Import Data" for 500 nodes and assign it to nodes_200
%nodes_200=CECnodes500TableToExcel;
%%

%Reducing from 1000s of nodes to 200 nodes
%for each scenario after computing velocity
%Input: Array with Flow Velocity and reduced_node set(ndoes_200)
%Output: Array with reduced number of nodes(200 or 500 ndoes)

%anomlay1_500 refers to leak node pressure

%Visualization: Click on the name of the variable
%from the Workspace section

%Reducing from 1000s of nodes to 200 nodes
%for each scenario
anomlay1_500_nodes=reducer(degree1_nodes,nodes_200);
anomlay11_500_nodes=reducer(degree11_nodes,nodes_200);
anomlay21_500_nodes=reducer(degree21_nodes,nodes_200);
anomlay31_500_nodes=reducer(degree31_nodes,nodes_200);
anomlay41_500_nodes=reducer(degree41_nodes,nodes_200);
anomaly_free_nodes_500=reducer(anomaly_free_nodes,nodes_200);

temps=[anomlay1_500_nodes.NodePressure,anomlay11_500_nodes.NodePressure,anomlay21_500_nodes.NodePressure,anomlay31_500_nodes.NodePressure,anomlay41_500_nodes.NodePressure];
new_arr=temps;

%Below code preprocesses data
%for AI purposes

%Software calucates for successive scenarios with maximum values.

for i=1:length(temps(:,1))
    maxs=max(temps(i,:));
    if anomaly_free_nodes_500(i,64).NodePressure>maxs
        maxs=anomaly_free_nodes_500(i,64).NodePressure;
        new_arr(i,1)=maxs-new_arr(i,1);
        new_arr(i,2)=maxs-new_arr(i,2);
        new_arr(i,3)=maxs-new_arr(i,3);
        new_arr(i,4)=maxs-new_arr(i,4);
        new_arr(i,5)=maxs-new_arr(i,5);
    else
        if temps(i,1) == maxs
            new_arr(i,1)=0;
            new_arr(i,2)=temps(i,1)-new_arr(i,2);
            new_arr(i,3)=temps(i,1)-new_arr(i,3);
            new_arr(i,4)=temps(i,1)-new_arr(i,4);
            new_arr(i,5)=temps(i,1)-new_arr(i,5);
            
        elseif temps(i,2)==maxs
            new_arr(i,1)=0;
            new_arr(i,2)=0;
%             new_arr(i,3)=new_arr(i,3)-temps(i,2);
%             new_arr(i,4)=new_arr(i,4)-temps(i,2);
%             new_arr(i,5)=new_arr(i,5)-temps(i,2);
            
        elseif temps(i,3)==maxs
            new_arr(i,1)=0;
            new_arr(i,2)=0;
            new_arr(i,3)=0;
%             new_arr(i,4)=new_arr(i,4)-temps(i,3);
%             new_arr(i,5)=new_arr(i,5)-temps(i,3);
        
        elseif temps(i,4)==maxs
            new_arr(i,1)=0;
            new_arr(i,2)=0;
            new_arr(i,3)=0;
            new_arr(i,4)=0;
%             new_arr(i,5)=new_arr(i,5)-temps(i,4);
            
        else
            new_arr(i,1)=0;
            new_arr(i,2)=0;
            new_arr(i,3)=0;
            new_arr(i,4)=0;
            new_arr(i,5)=0;
        
        end
    end
end

%%
%Preparing and storing the Pressure database for SVM and ANN
%Storing data from Scenario 1 to Scenario 5 for each node in a serial order of the array
%Eg: saved(1,5)=Scenario 1 to scenario 5 for node 1,
%    saved(6,10)=Scenario 1 to scenario 5 for node 2 and so on.

saved = zeros(length(anomlay1_500_nodes.NAME)*5,1);
j=1;
for i=1:5:1020
    saved(i)=new_arr(j,1);
    saved(i+1)=new_arr(j,2);
    saved(i+2)=new_arr(j,3);
    saved(i+3)=new_arr(j,4);
    saved(i+4)=new_arr(j,5);
    j=j+1;
end

%Saving modified data to Mymatrix_node.txt
%Use Mymatrix_node.txt as input for AI code

fid = fopen('Mymatrix_32node.txt','wt');
for ii = 1:size(saved,1)
    fprintf(fid,'%g\t',saved(ii,:));
    fprintf(fid,'\n');
end
fclose(fid);

%Savinig the names of the reduced_node_set for pressure
fid = fopen('Pressure_node_names.txt','wt');
for ii = 1:size(anomlay1_500_nodes.NAME,1)
    fprintf(fid,'%s\n',anomlay1_500_nodes.NAME(ii,:));
end

%Savinig the pressures of anomaly free
fid = fopen('Pressure_anomaly_free.txt','wt');
for ii = 1:size(anomaly_free_nodes_500.NodePressure,1)
    fprintf(fid,'%s\n',anomaly_free_nodes_500.NodePressure(ii,:));
end

function df_new=reducer(input_arr,nodes_200)
    unique_nodes=unique(nodes_200.NAME);
    temp1=find(ismember(input_arr.NAME,unique_nodes));
    final= input_arr(temp1,:);
    df_new=final;
end
%%



