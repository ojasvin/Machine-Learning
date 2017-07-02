function [ftheta1,ftheta2]=optimizetheta (Train,lrate,nodesin,nodeshid,nodesout)
    epochs=1000;
    Row=size(Train,1);
    s=zeros(size(Train,1),nodesout);
    Col=size(Train,2);
    theta1=rand(nodesin+1,nodeshid);
    theta2=rand(nodeshid+1,nodesout);
    for m=1:epochs
        y1(:,2:nodesin+1)=Train(:,1:nodesin);
        y1(:,1)=ones(size(Train,1),1);
        y1=y1';
        v2=theta1'*y1;
        y2=sigmoid(v2);
        y2=y2';
        y2(:,2:nodeshid+1)=y2(:,1:nodeshid);
        y2(:,1)=ones(size(Train,1),1);
        y2=y2';
        v3=theta2'*y2;
        y3=sigmoid(v3);%%%3*120
        %%%%Backward propagation%%%%%
%         for i=1:size(Train,1)
%             if Train(i,Col)==1
%                 s(i,1:3)=[1 0 0];
%             elseif Train(i,Col)==2
%                 s(i,1:3)=[0 1 0];
%             else s(i,1:3)=[0 0 1];
%             end
%         end
   
        for j=1:nodesout
            for i=1:size(Train,1)
                abc=zeros(1,nodesout);
                if Train(i,Col)==j
                    abc(1,j)=1;
                    s(i,1:nodesout)=abc;
                end
            end
        end
        error=s'-y3;%%%3*120
        delta3=error.*y3.*(1-y3);%%%%3*120
        wowy2(1:nodeshid,1:Row)=y2(2:nodeshid+1,:);
        delta2=(theta2(2:nodeshid+1,:)*delta3).*(wowy2.*(1-wowy2));%%%6*120
        %%%weight updation%%%%
        theta2=theta2+lrate*y2*delta3';   %%%7*3
        theta1=theta1+lrate*y1*delta2';   %%%5*6
        y1=y1';
        y2=y2';
        y3=y3';
    end
    ftheta1=theta1;
    ftheta2=theta2;
end
