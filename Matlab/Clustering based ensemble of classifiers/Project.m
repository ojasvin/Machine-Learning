
clear;
close all;
tic
%%%%Wine data Highest accuracy is 97.22 epochs 5000 lrate 0.05 hidnodes 11
% B = load('wine.data');
% Col=size(B,2);
% Row=size(B,1);
% A(:,1:Col-1)=B(:,2:Col);
% A(:,Col)=B(:,1);

%%%%%Corners.txt%%%%%%%%%%%%%%%%
%%%%Combined two diagnol classes to know that does the clustering actually
%%%%help the cause or not it did help it
%  A=load('Corners.txt');
%  Row=size(A,1);
%  Col=size(A,2);
%  A(:,Col)=A(:,Col)+1;
%  for i=1:Row
%      if A(i,Col)==4
%          A(i,Col)=3;
%      elseif A(i,Col)==2
%          A(i,Col)=1;
%      else
%          A(i,Col)=A(i,Col);
%      end
%  end
%  for i=1:Row
%      if A(i,Col)==3
%          A(i,Col)=2;
%      end
%  end

%%%%%%Outliers %%%%%
%A=load('Outliers.txt');
%Row=size(A,1);
%Col=size(A,2);
%A(:,Col)=A(:,Col)+1;
%for i=1:Row
 %   if A(i,Col)==4
  %      A(i,Col)=1;
   % elseif A(i,Col)==3
    %    A(i,Col)=2;
   % else
    %    A(i,Col)=A(i,Col);
    %end
%end


%%%Balance scale data acc highest is 93.65 epochs 5000 and lrate 0.05 nidnodes 11 
% B = load('balance-scale.data');
% Col=size(B,2);
% Row=size(B,1);
% A(:,1:Col-1)=B(:,2:Col);
% A(:,Col)=B(:,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%IRIS DATASET hidden nodes=11%%%%%%%%%%%%%%%
filename='iris.txt';
A= load(filename);
Col=size(A,2);
Row=size(A,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Glass data set accuracy is 65.11 to 81.39 lrate 0.05 epochs 5000 hid nodes 11%%%
% filename='glass.data';
% A= load(filename);
% Col=size(A,2);
% Row=size(A,1);
% for i=1:size(A,1)
%     if A(i,Col)==7
%         A(i,Col)=4;
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%Ionosphere data hidden nodes 11%%%%%
% B = load('ionosphere.data');
% Col=size(B,2);
% Row=size(B,1);
% A(:,1:Col-1)=B(:,2:Col);
% A(:,Col)=B(:,1);
% for i=1:size(A,1)   
%     A(i,Col)=A(i,Col)+1;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%Breast Cancer Wiscosin hidden nodes 14%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


hid=11;%%%choose no of hidden nodes here
U=unique(A(:,Col));
out=size(U,1);%No of output nodes
%%Data Normalization
for i=1:size(A,1)
    for j=1:Col-1
        if max(A(:,j))~=0 || min(A(:,j))~=0
            A(:,j)=(A(:,j)-min(A(:,j)))/(max(A(:,j))-min(A(:,j)));
        end
    end
end
%%%%gscatter(A(:,1),A(:,2),A(:,3));
for i=1:out
    c(i,1)=1;
end
E=cell(out,1);%Store entire classes
F=cell(out,1);%Storing only the training part of it
groups=cell(out,1);
for j=1:out
    for i=1:size(A,1)    
        if A(i,Col)==j
            E{j}(c(j,1),:)=A(i,:);%%%Store each target class
            c(j,1)=c(j,1)+1;
        end
    end
end
for i=1:out
    E{i}=E{i}(randperm(size(E{i},1)),:);
end
for i=1:out
    F{i}(1:(size(E{i},1)-round(size(E{i},1)*0.2)),:)=E{i}(1:(size(E{i},1)-round(size(E{i},1)*0.2)),:);%%%Prepare the training data of each target class 
    F{i}=F{i}(randperm(size(F{i},1)),:);
end

ftest(1:round(size(E{1},1)*0.2),:)=E{1}(1+size(E{1},1)-round(size(E{1},1)*0.2):size(E{1},1),:);%%Create testing data
for i=2:out
    ftest=[ftest ; E{i}(1+size(E{i},1)-round(size(E{i},1)*0.2):size(E{i},1),:)];
end

ftest=ftest(randperm(size(ftest,1)),:);
   
sum=0;
for i=1:out
    [K(i,1),groups{i}]=kmeansclusterring(F{i});
    sum=sum+K(i,1);
end
%%%%For printing 2d data
% for i=1:out
%     gscatter(F{i}(:,1),F{i}(:,2),F{i}(:,3));
%     
% end
%
% c=1;
% for i=1:K(1,1)
%     for j=1:size(groups{1}{i},2)
%         C(c,:)=F{1}(groups{1}{i}(1,j),:);
%         C(c,3)=i;
%         c=c+1;
%     end  
% end
% for i=1:K(2,1)
%     for j=1:size(groups{2}{i},2)
%         C(c,:)=F{2}(groups{2}{i}(1,j),:);
%         C(c,3)=i+K(1,1);
%         c=c+1;
%    end
% end
% gscatter(C(:,1),C(:,2),C(:,3));
M1=cell(sum,1);
M2=cell(sum,1);
a=0;
for j=1:out
   for i=1:K(j,1)
       a=a+1;
       if groups{j}{i}~=0%%%So that no empty clusters 
            OS=Trainprep(j,out,F);%%Prepare the training part of it by using 25 per of other target classes
            [ftheta1,ftheta2]=mlp(out,hid,F{j},OS,groups{j}{i});%%%Multi layer  perceptron 
            M1{a}=ftheta1;
            M2{a}=ftheta2;           
       end
    end
    
end
for i=1:sum
    if (size(M1{i},1)~=0)&&(size(M2{i},1)~=0)
        class(:,i)=test(ftest,M1{i},M2{i});%%%Prediction of each of classifiers
    end
end
count=0;


for j=1:size(ftest,1)
    if ftest(j,Col)==mode(class(j,:))
        count=count+1;
    end
end
for j=1:size(ftest,1)
    predict(j,1)=mode(class(j,:));
end
acc=(count/size(ftest,1))*100%%%Accuracy
conf=confusionmatrix(predict,out,ftest(:,Col));%%%%Confusion matrix 
toc
