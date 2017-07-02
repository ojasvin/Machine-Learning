function [ftheta1,ftheta2]=optimizetheta(Train,lrate,nodesin,nodeshid,nodesout)
    epochs=5000;
    Row=size(Train,1);
    Col=size(Train,2);    
    theta1=rand(nodesin+1,nodeshid);%5*5
    theta2=rand(nodeshid+1,nodesout);%6*1
    for m=1:epochs
        %%%Feedforwardstep%%%
        y1(:,2:Col)=Train(:,1:Col-1);
        y1(:,1)=ones(size(Train,1),1);
        y1=y1';%5*4
        v2=theta1'*y1;
        y2=sigmoid(v2);
        y2=y2';%4*5
        y2(:,2:nodeshid+1)=y2(:,1:nodeshid);
        y2(:,1)=ones(size(Train,1),1);%4*6
        y2=y2';%6*4
        v3=theta2'*y2;%1*4
        y3=sigmoid(v3);%%%1*4
        %%%%Backward propagation%%%%%
        for j=1:nodesout
            for i=1:size(Train,1)
                abc=zeros(1,nodesout);
                if Train(i,Col)==j
                    abc(1,j)=1;
                    s(i,1:nodesout)=abc;
                end
            end
        end
        error=s'-y3;%%%
        delta3=error.*y3.*(1-y3);%%%%
        wowy2(1:nodeshid,:)=y2(2:nodeshid+1,:);
        delta2=(theta2(2:nodeshid+1,:)*delta3).*(wowy2.*(1-wowy2));%%%
        %%%weight updation%%%%
        theta2=theta2+lrate*y2*delta3';   %%%
        theta1=theta1+lrate*y1*delta2';   %%%
        y1=y1';
        y2=y2';
        y3=y3';
    end
    ftheta1=theta1;
    ftheta2=theta2;
 end


