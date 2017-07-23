function result =test(ftest,ftheta1,ftheta2)
        result=zeros(size(ftest,1),1);
        nodesin=size(ftheta1,1)-1;
        nodeshid=size(ftheta1,2);
        nodesout=size(ftheta2,2);
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
        result(i,1)=maxindex;%%%%Find the result after testing
    end
end
    
    
    

    
    
    
    
    