function cmatrix = confusionmatrix(predict,out,ftest)
cmatrix=zeros(out,out);
for i=1:out
    for j=1:size(predict,1)
        if ftest(j,1)==i
            cmatrix(i,predict(j,1))=cmatrix(i,predict(j,1))+1;
        end
    end
end
cmatrix%%%%Prints the confusion matrix
end