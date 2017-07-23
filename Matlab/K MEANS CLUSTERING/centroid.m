function c= centroid (A,X)
for m=1:size(A,2)-1
    c(m)=0;
    for i=1:size(X,2)
        c(m)=c(m)+A(X(1,i),m);
    end
    c(m)=c(m)/size(X,2);
end
end