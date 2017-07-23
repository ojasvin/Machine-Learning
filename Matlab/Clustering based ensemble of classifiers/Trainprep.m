function[B]= Trainprep (j,out,F)
  N=cell(out-1,1);
  c=2;
    if j~=1
        N{1}=F{1};
        N{1}=N{1}(randperm(round(size(N{1},1)*0.25)),:);%%%%To introduce 25 of training data of other classes as well
        for i=2:out
            if i~=j
                N{c}=F{i};
                N{c}=N{c}(randperm(round(size(N{c},1)*0.25)),:);
                c=c+1;
            end
        end
    else
        N{1}=F{2};
        N{1}=N{1}(randperm(round(size(N{1},1)*0.25)),:);
        for i=3:out
            N{c}=F{i};
            N{c}=N{c}(randperm(round(size(N{c},1)*0.25)),:);
            c=c+1;
        end
    end
    B=N{1};
    for i=2:out-1
        B=[B;N{i}];
    end
end
