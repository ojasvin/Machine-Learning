function [I,groups2]=kmeansclusterring (A)


Col=size(A,2);
Dim=size(A);
Row=size(A,1);
cluster=zeros(Row,1);
sum=zeros(1,Col-1);
 %Selecting the row number.
%disp(Selection);
% disp(A);
%--------------------------- k random centres ----------------------------%
Dunn(1)=0;
DB(1)=999;
for K=2:10
    dist=zeros(K,1);
    Selection=rand(1,K);
    Selection=Selection*Dim(1,1);
    Selection=ceil(Selection);
    for k=1:K
        mu(k,1:Col-1)=A(Selection(k),1:Col-1);
    end
    c=zeros(K,1);
    for l=1:5000
        %%%%%%%%%%%%%%%%%Cluster Assignment Step%%%%%%%%%%%%%%%%%%%%%
        for j=1:Row
            x=A(j,:);
            for k=1:K
                for m=1:Col-1
                    dist(k,1)=dist(k,1)+((x(1,m)-mu(k,m))^2);              
                end
            end
            minimum=min(dist);
            row=find(minimum==dist);
            dist(:,1)=0;

            cluster(j,1)=row(1);       
        end
        c=zeros(K,Col-1);
        %%%%%%%%%%%%%New Centroid Step%%%%%%%%%%%%%%%%%%%%
        for k=1:K       
             sum(1,:)=zeros(1,Col-1);
             for m=1:Col-1
                 for i=1:Row
                    x=A(i,:);
                    if cluster(i,1)==k
                        sum(1,m)=sum(1,m)+x(m);
                        c(k,m)=c(k,m)+1;
                    end
                 end
                 mu(k,m)=sum(1,m)/c(k,m);%%%New cluster centroid
                 c(k,m)=0;
             end
        end  
    end
    %%%%%%%%%%Clustering is done now%%%%%%%%%%%%%%%%%%%%%5
    for k=1:K
        groups{k}=[];    
    end
    c=zeros(k,1);
    for k=1:K    
        for i=1:Row
            if cluster(i,1)==k
               groups{k}=[groups{k} i]; 
            end
        end
    end
%     fprintf('For no of clusters = %d \n',K);
%     for k=1:K
%        fprintf('The size of cluster %d is\n',k);
%        disp(size(groups{k},2));
%     end
%     fprintf('\n\n\n');
    
    %%%%%%%%%%%%%CLUSTER VALIDITY INDICES DUNN AND DB INDEX%%%%%%%%%
    %%%%%%%%%%%Cluster Centroid%%%%%%%%%%    
    for i=1:K
        centre(i,1:Col-1)=centroid(A,groups{i});
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%Dunn index%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    minimum=10000;
    for i=1:K
        for j=1:K
            dist(i,j)=10000;
            for l1=1:size(groups{i},2)
                for l2=1:size(groups{j},2)
                    if i~=j
                        minimum=min(minimum,distance(A,groups{i}(l1),groups{j}(l2)));
                        dist(i,j)=min(dist(i,j),distance(A,groups{i}(l1),groups{j}(l2)));%%%%%%%%Distance between two clusters%%%%%% 
                    end
                end
            end
        end
    end
    maximum=-9999;
    for i=1:K
        for l1=1:size(groups{i},2)
            for l2=1:size(groups{i},2)
                if groups{i}(l1)~=groups{i}(l2)
                      maximum=max(maximum,distance(A,groups{i}(l1),groups{i}(l2)));
                end
            end
        end
    end
    Dunn(K)=minimum/maximum;
    %%%%%%%DB INDEX%%%%%%%%%%%%%%%%%%%%%%
    for i=1:K
        s(i)=-1;
        for l1=1:size(groups{i},2)
            s(i)=max(s(i),distancesi(A,groups{i}(l1),centre(i,:)));
        end
    end
    %%%%%%%%%%%%%R(I,J)%%%%%%%%%%%%%%%%%%%%%
    for i=1:K
        for j=1:K
            if i~=j
                r(i,j)=(s(i)+s(j))/dist(i,j);
            end
        end
    end
    %%%%%%%%%%%%%R(I)%%%%%%%%%%%%%%%
   %%% fprintf('\n\n');
    
    for i=1:K
        r1(i)=-120;
        for j=1:K
            if j~=i
%                 fprintf('\n\n');
                r1(i)=max(r1(i),r(i,j));
            end
        end
      
    end
    DB(K)=0;
    for i=1:K
        DB(K)=DB(K)+r1(i);
    end
    DB(K)=(1/K)*(DB(K));   
    
end
[M,I]=max(Dunn);
dist=zeros(I,1);
Selection=rand(1,I);
Selection=Selection*Dim(1,1);
Selection=ceil(Selection);
for k=1:I
    mu(k,1:Col-1)=A(Selection(k),1:Col-1);
end
c=zeros(I,1);
for l=1:5000
    %%%%%%%%%%%%%%%%%Cluster Assignment Step%%%%%%%%%%%%%%%%%%%%%
    for j=1:Row
        x=A(j,:);
        for k=1:I
            for m=1:Col-1
                dist(k,1)=dist(k,1)+((x(1,m)-mu(k,m))^2);              
            end
        end
        minimum=min(dist);
        row=find(minimum==dist);
        dist(:,1)=0;

        cluster(j,1)=row(1);       
    end
    c=zeros(I,Col-1);
    %%%%%%%%%%%%%New Centroid Step%%%%%%%%%%%%%%%%%%%%
    for k=1:I       
         sum(1,:)=zeros(1,Col-1);
         for m=1:Col-1
             for i=1:Row
                x=A(i,:);
                if cluster(i,1)==k
                    sum(1,m)=sum(1,m)+x(m);
                    c(k,m)=c(k,m)+1;
                end
             end
             mu(k,m)=sum(1,m)/c(k,m);%%%New cluster centroid
             c(k,m)=0;
         end
    end  
end
%%%%%%%%%%Clustering is done now%%%%%%%%%%%%%%%%%%%%%5
for k=1:I
    groups2{k}=[];    
end
c=zeros(k,1);
for k=1:I    
    for i=1:Row
        if cluster(i,1)==k
           groups2{k}=[groups2{k} i]; 
        end
    end
end


end