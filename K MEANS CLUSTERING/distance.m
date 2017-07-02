function dist = distance(A,i,j)
dist=0;

for m=1:size(A,2)-1
    dist=dist+(A(i,m)-A(j,m))^2;
end
dist=sqrt(dist);
end
