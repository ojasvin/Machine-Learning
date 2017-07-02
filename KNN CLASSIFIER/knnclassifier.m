
function accuracy = knnclassifier(train,test,k) 
Col=size(train,2);
dist=zeros(size(train,1),1);
for i = 1: size(test)
    x=test(i,:);
    for j=1:Col-1
        for sh=1:size(train)
            dist(sh,1)=dist(sh,1)+(x(j)-train(sh,j)).^2;  
        end
    end
    dist(:,1)=sqrt(dist(:,1)); 
    dist(:,2)=train(:,Col);
    sortd=sortrows(dist,1);
    finalclass(i,1)=mode(sortd(1:k,2));
end


error=finalclass(:,1)-test(:,Col);
accuracy=(size(error)-nnz(error))/size(error);
end
