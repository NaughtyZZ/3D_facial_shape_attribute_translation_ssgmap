clear;
clc;

load output.mat
load norm_parameters.mat 
[~,normal,texture,triList]=plyRead('regisered_data.ply');


for i=1:size(shape,1)
shape2=squeeze(shape(i,1,:,:));
shape2=shape2'.*repmat(divide_factor,1,length(shape2))+repmat(minima_norm,1,length(shape2));
outFileName=strcat('./out_ply/',int2str(i),'.ply');
normal2=vertexNormal(shape2',triList');
normal2=normal2';
normal3=normal;
index=isnan(normal2);
index=index(1,:)&index(2,:)&index(3,:);
normal2(:,index)=normal3(:,index);
writePlyFile(outFileName,shape2,normal2,texture,triList);
end