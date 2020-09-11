%Name of the files uploaded
degree41_pipes=NYUAnamolyDataZeroDegPipesLeak41;
degree31_pipes=NYUAnamolyDataZeroDegPipesLeak31;
degree21_pipes=NYUAnamolyDataZeroDegPipesLeak21;
degree11_pipes=NYUAnamolyDataZeroDegPipesLeak11;
degree1_pipes=NYUAnamolyDataZeroDegPipesLeak1;
anomaly_free_pipes=NYUAnamolyDataZeroDegPipes;
nodes_200=CECnodes200TableToExcel;

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
            
saved = zeros(1025,1);
j=1;
for i=1:5:1025
    saved(i)=new_arr(j,1);
    saved(i+1)=new_arr(j,2);
    saved(i+2)=new_arr(j,3);
    saved(i+3)=new_arr(j,4);
    saved(i+4)=new_arr(j,5);
    j=j+1;
end


fid = fopen('Mymatrix.txt','wt');
for ii = 1:size(saved,1)
    fprintf(fid,'%g\t',saved(ii,:));
    fprintf(fid,'\n');
end
fclose(fid);

    


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




