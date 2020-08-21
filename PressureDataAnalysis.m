%Name of the files uploaded
degree41_nodes=NYUAnamolyDataZeroDegNodesLeak41;
degree31_nodes=NYUAnamolyDataZeroDegNodesLeak31;
degree21_nodes=NYUAnamolyDataZeroDegNodesLeak21;
degree11_nodes=NYUAnamolyDataZeroDegNodesLeak11;
degree1_nodes=NYUAnamolyDataZeroDegNodesLeak1;
anomaly_free_nodes=NYUAnamolyDataZeroDegNodes;
nodes_200=CECnodes200TableToExcel;

anomlay1_500_nodes=reducer(degree1_nodes,nodes_200);
anomlay11_500_nodes=reducer(degree11_nodes,nodes_200);
anomlay21_500_nodes=reducer(degree21_nodes,nodes_200);
anomlay31_500_nodes=reducer(degree31_nodes,nodes_200);
anomlay41_500_nodes=reducer(degree41_nodes,nodes_200);
anomaly_free_nodes_500=reducer(anomaly_free_nodes,nodes_200);

temps=[anomlay1_500_nodes.NodePressure,anomlay11_500_nodes.NodePressure,anomlay21_500_nodes.NodePressure,anomlay31_500_nodes.NodePressure,anomlay41_500_nodes.NodePressure];
new_arr=temps;

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
            
saved = zeros(1020,1);
j=1;
for i=1:5:1020
    saved(i)=new_arr(j,1);
    saved(i+1)=new_arr(j,2);
    saved(i+2)=new_arr(j,3);
    saved(i+3)=new_arr(j,4);
    saved(i+4)=new_arr(j,5);
    j=j+1;
end


fid = fopen('Mymatrix_node.txt','wt');
for ii = 1:size(saved,1)
    fprintf(fid,'%g\t',saved(ii,:));
    fprintf(fid,'\n');
end
fclose(fid);

    

function df_new=reducer(input_arr,nodes_200)
    unique_nodes=unique(nodes_200.NAME);
    temp1=find(ismember(input_arr.NAME,unique_nodes));
    final= input_arr(temp1,:);
    df_new=final;

end




