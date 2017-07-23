function [ftheta1,ftheta2] = mlp(nodesout,nodeshid,data,B,groups)
Col=size(data,2);
for i=1:size(groups,2)
    Train(i,1:Col)=data(groups(1,i),1:Col);%%%prepares the training data
end
Train=[Train;B(randperm(size(B,1)),:)];%%%Also introduces training data of other classes
Col=size(Train,2);
nodesin=Col-1;
lrate=0.05;%%%learning rate
[ftheta1,ftheta2]=optimizetheta(Train,lrate,nodesin,nodeshid,nodesout);%%To get correct theta after backpropagation
     
end
