function dist = distance(A,i,y)
dist=0;

for m=1:size(A,2)-1
    dist=dist+(A(i,m)-y(1,m))^2;
end
dist=sqrt(dist);
end