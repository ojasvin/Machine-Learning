clear;
close all;

%%%%%%%IRIS DaTA SET%%%%%%%%%%%%
filename='iris.txt';%%%IRIS DATA WE WILL HAVE 3 LAYERS 1 INPUT LAYER 1 HIDDEN LAYER I OUTPUT LAYER
A = load(filename);
Col=size(A,2);
Row=size(A,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% B = load('balance-scale.data');
% Col=size(B,2);
% Row=size(B,1);
% A(:,1:Col-1)=B(:,2:Col);
% A(:,Col)=B(:,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% B = load('wine.data');
% Col=size(B,2);
% Row=size(B,1);
% A(:,1:Col-1)=B(:,2:Col);
% A(:,Col)=B(:,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% A=load('breast-cancer-wisconsin.data');
% Col=size(A,2);
% Row=size(A,1);
% for i=1:size(A,1)
%     for j=1:Col
%         if A(i,Col)==-5
%             A(i,:)=[];
%         end
%     end
% end
% for i=1:size(A,1)
%     A(i,Col)=A(i,Col)/2;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% filename='glass.data';
% A= load(filename);
% Col=size(A,2);
% Row=size(A,1);
% for i=1:size(A,1)
%     if A(i,Col)==7
%         A(i,Col)=4;
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% B = load('ionosphere.data');
% Col=size(B,2);
% Row=size(B,1);
% A(:,1:Col-1)=B(:,2:Col);
% A(:,Col)=B(:,1);
% for i=1:size(A,1)   
%     A(i,Col)=A(i,Col)+1;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%5555555

for i=1:size(A,1)
    for j=1:Col-1
        if max(A(:,j))~=0 || min(A(:,j))~=0
            A(:,j)=(A(:,j)-min(A(:,j)))/(max(A(:,j))-min(A(:,j)));
        end
    end
end

U=unique(A(:,Col));
out=size(U,1);


A = A(randperm(Row),:);
X = A(1:round(0.8*Row),:); %0.8 because 80 percent of data used as train and 20 percent used as test
ftest=A(round(0.8*Row)+1:Row,:);
nodesin=Col-1;
nodeshid=5;
nodesout=out;
lrate=0.05;
for l=1:10
    [ftheta1,ftheta2]=optimizetheta(X,lrate,nodesin,nodeshid,nodesout);
    %%%%ftheta1 size (5*6) ftheta2 size (7*3)  
    y1(:,2:nodesin+1)=ftest(:,1:nodesin);
    y1(:,1)=ones(size(ftest,1),1);
    v2=ftheta1'*y1';
    y2=sigmoid(v2);
    y2=y2';
    y2(:,2:nodeshid+1)=y2(:,1:nodeshid);
    y2(:,1)=ones(size(ftest,1),1);
    v3=ftheta2'*y2';
    y3=sigmoid(v3);%%%%3*30
    y3=y3';
    count=0;
    for i=1:size(ftest,1)
        x=y3(i,:);
        max=x(:,1);
        maxindex=1;
        for j=2:nodesout
            if x(:,j)>max
               max=x(:,j);
               maxindex=j;
            end
        end
        if maxindex==ftest(i,Col)
            count=count+1;
        end
    end
    acc=count/size(ftest,1);
    if l==1
        minimumacc=acc*100;
        maximumacc=acc*100;
    end
    if minimumacc>acc*100
       minimumacc=acc*100;
    end
    if maximumacc<acc*100
       maximumacc=acc*100;
    end
    s(l)=acc*100;
end
sum=0;
for i=1:10
    sum=sum+s(i);
end
% fprintf('Accuracy is %f \n',acc*100);
fprintf('Average Accuracy is %f\n\n',sum/10);  
fprintf('Maximum Accuracy is %f\n\n',maximumacc);
fprintf('Minimum Accuracy is %f\n\n',minimumacc);