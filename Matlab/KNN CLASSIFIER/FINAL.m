clear;
close all;

%Trainfile = 'mnist_train.csv';
%Test='mnist_test.csv';
filename='iris.txt';
%filename='creditcard.xlsx';
%filename='Half_kernel.txt';
%filename='Image3.txt';
%filename='voice.txt';
%filename='DoubleMoon1.txt';
A = load(filename);
%A= load(Trainfile);
Col=size(A,2);
%Now the data set is ready

n=size(A,1);

A = A(randperm(n),:);
X=A(1:0.8*n,:);%0.8 because 80 percent of data used as train and 20 percent used as test
%X=A(1:5000,:);
ftest=A(0.8*n+1:n,:);
%ftest=load(Test);
fold=5;

for K = 1 : 10 %for 10 different odd values
    c=0;
    s=0;
         
    for chunk = 1:fold%parallel processing
        chunksize=size(X,1)/fold;
        x = (chunk - 1) * chunksize + 1;
        y = chunk * chunksize;
        test=X(x:y,:);
        if chunk == 1
            train=X(y+1:end,:);
        
        elseif chunk == fold
            train=X(1:x-1,:);
       
        else
            train=X([1,x-1:y+1,end],:);             
        end
       
        curracc=knnclassifier(train,test,2*K-1); %calling knnclassifier function
        s=s+curracc;
        c=c+1; 
     
    end
    acc(K)=s/c;
end             
fprintf('The accuracy in training data is:: \n');
for i=1:10
    fprintf('For K=%d %f\n',2*i-1,100*acc(i));
end
M=find(acc==max(acc(2:end)));%Except K=1 because K=1 is 1 NN WHICH ON TRAIN WILL GIVE OVERFIT
fprintf('The best K value is %d ',2*M-1);
fprintf('\n');            


%%%% For Test Data    
fprintf('The accuracy in test data is:: \n');
curracc2=knnclassifier(X,ftest,2*M-1);%Here k=2*M-1 is sent and We get accuracy for our test data
disp(curracc2*100);

