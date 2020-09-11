%Flow data pre processing File
%Click on "Import Data" to import the Flow files for the 5 scenarios

%Name of the files to be uploaded
%Eg: NYUAnamolyDataZeroDegPipesLeak41 refers to file with leak value of 41
degree41_pipes=NYUAnamolyDataZeroDegPipesLeak41;
degree31_pipes=NYUAnamolyDataZeroDegPipesLeak31;
degree21_pipes=NYUAnamolyDataZeroDegPipesLeak21;
degree11_pipes=NYUAnamolyDataZeroDegPipesLeak11;
degree1_pipes=NYUAnamolyDataZeroDegPipesLeak1;
anomaly_free_pipes=NYUAnamolyDataZeroDegPipes;
%Name of file with 200 nodes
nodes_200=CECnodes200TableToExcel;

%%
%Note to user
%For reducing to 500 nodes
%"Import Data" for 500 nodes and assign it to nodes_200
%nodes_200=CECnodes500TableToExcel;
%%

%pipe_velocity_calc used to calculate flow velocity
%reducer used to reduce number of nodes to 200
%Input: Array with Flow and Anomaly free array
%Output: Array with Flow Velocity

%Reducing from 1000s of nodes to 200 nodes
%for each scenario after computing velocity
%Input: Array with Flow Velocity and reduced_node set(ndoes_200)
%Output: Array with reduced number of nodes(200 or 500 ndoes)

%anomlay1_500 refers to leak node flow velocity

%Visualization: Click on the name of the variable
%from the Workspace section

val=pipe_velocity_calc(degree1_pipes,anomaly_free_pipes);
anomlay1_500=reducer(val,nodes_200);
val=pipe_velocity_calc(degree11_pipes,anomaly_free_pipes);
anomlay11_500=reducer(val,nodes_200);
val=pipe_velocity_calc(degree21_pipes,anomaly_free_pipes);
anomlay21_500=reducer(val,nodes_200);
val=pipe_velocity_calc(degree31_pipes,anomaly_free_pipes);
anomlay31_500=reducer(val,nodes_200);
val=pipe_velocity_calc(degree41_pipes,anomaly_free_pipes);
anomlay41_500=reducer(val,nodes_200);
val=pipe_velocity_calc(anomaly_free_pipes,anomaly_free_pipes);
anomaly_free_pipes_500=reducer(val,nodes_200);

temps=[anomlay1_500.VELOpipeFPS,anomlay11_500.VELOpipeFPS,anomlay21_500.VELOpipeFPS,anomlay31_500.VELOpipeFPS,anomlay41_500.VELOpipeFPS];
new_arr=temps;

%Below code preprocesses data
%for AI purposes

%Software calucates for successive scenarios with minimum values.


for i=1:length(temps(:,1))
    mins=min(temps(i,:));
    if anomaly_free_pipes_500(i,136).VELOpipeFPS<mins
        mins=anomaly_free_pipes_500(i,136).VELOpipeFPS;
        new_arr(i,1)=new_arr(i,1)-mins;
        new_arr(i,2)=new_arr(i,2)-mins;
        new_arr(i,3)=new_arr(i,3)-mins;
        new_arr(i,4)=new_arr(i,4)-mins;
        new_arr(i,5)=new_arr(i,5)-mins;
    else
        if temps(i,1) == mins
            new_arr(i,1)=0;
            new_arr(i,2)=new_arr(i,2)-temps(i,1);
            new_arr(i,3)=new_arr(i,3)-temps(i,1);
            new_arr(i,4)=new_arr(i,4)-temps(i,1);
            new_arr(i,5)=new_arr(i,5)-temps(i,1);
            
        elseif temps(i,2)==mins
            new_arr(i,1)=0;
            new_arr(i,2)=0;
%             new_arr(i,3)=new_arr(i,3)-temps(i,2);
%             new_arr(i,4)=new_arr(i,4)-temps(i,2);
%             new_arr(i,5)=new_arr(i,5)-temps(i,2);
            
        elseif temps(i,3)==mins
            new_arr(i,1)=0;
            new_arr(i,2)=0;
            new_arr(i,3)=0;
%             new_arr(i,4)=new_arr(i,4)-temps(i,3);
%             new_arr(i,5)=new_arr(i,5)-temps(i,3);
        
        elseif temps(i,4)==mins
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
%Preparing and storing the Flow database for SVM and ANN
%Storing data from Scenario 1 to Scenario 5 for each node in a serial order of the array
%Eg: saved(1,5)=Scenario 1 to scenario 5 for node 1,
%    saved(6,10)=Scenario 1 to scenario 5 for node 2 and so on.

saved = zeros(length(anomlay1_500.NAME)*5,1);
j=1;
for i=1:5:1025
    saved(i)=new_arr(j,1);
    saved(i+1)=new_arr(j,2);
    saved(i+2)=new_arr(j,3);
    saved(i+3)=new_arr(j,4);
    saved(i+4)=new_arr(j,5);
    j=j+1;
end

%Saving data to Mymatrix.txt
%Use Mymatrix.txt as input for AI code

fid = fopen('Mymatrix.txt','wt');
for ii = 1:size(saved,1)
    fprintf(fid,'%g\t',saved(ii,:));
    fprintf(fid,'\n');
end
fclose(fid);


fid = fopen('Flow_pipe_names.txt','wt');
for ii = 1:size(anomlay1_500.NAME,1)
    fprintf(fid,'%s\n',anomlay1_500.NAME(ii,:));
end
fclose(fid);
%%

function pipedf = pipe_velocity_calc(pipedf,anomaly_free_pipes)
    zeros_arr=zeros(length(pipedf.NAME),1);
    in2_ft2 = 0.0005787;
    final_pipedf=[pipedf.NAME,pipedf.FacilityFromNodeName,pipedf.FacilityToNodeName,pipedf.FacilityFlowAbsolute,pipedf.PipeDiameter,zeros_arr,zeros_arr,zeros_arr,zeros_arr,zeros_arr];
    for i=1:length(zeros_arr)
        if contains(final_pipedf(i,1),anomaly_free_pipes.NAME)
            temp= find((pipedf.NAME==final_pipedf(i,1))==1);
            final_pipedf(i,6)=pipedf(temp,91).FacilityFlowAbsolute;
            final_pipedf(i,7)=pipedf(temp,95).PipeDiameter;
        end
    end
    elem_AREApipeFT2 = ((final_pipedf(:,7).double).^2/4)*pi*in2_ft2;
    elem_VELOpipeFPS = ((final_pipedf(:,6).double)./elem_AREApipeFT2*(1000/3600));
    final_pipedf(:,9) = elem_AREApipeFT2;
    final_pipedf(:,10) = elem_VELOpipeFPS;
    pipedf.AREApipeFT2 = elem_AREApipeFT2;
    pipedf.VELOpipeFPS = elem_VELOpipeFPS;
    
end

function df_new=reducer(input_arr,nodes_200)
    unique_nodes=unique(nodes_200.NAME);
    temp1=find(ismember(input_arr.FacilityFromNodeName,unique_nodes));
    reduced_nodeArr= input_arr(temp1,:);
    temp1=find(ismember(input_arr.FacilityToNodeName,unique_nodes));
    reduced_nodeArr1 = input_arr(temp1,:);
    r=setdiff(reduced_nodeArr(:,1),reduced_nodeArr1(:,1));
    temp1=find(~ismember(reduced_nodeArr.NAME,r.NAME));
    final=reduced_nodeArr(temp1,:);
    df_new=final;

end
%%






