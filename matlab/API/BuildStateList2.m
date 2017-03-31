function [ states ] = BuildStateList2(N_grid)
%BuildStateList builds a state list from a state matrix

% state discretization for the mountain car problem



x1   = linspace(0,1,N_grid);
x2   = linspace(0,1,N_grid);

I=size(x1,2);
J=size(x2,2);



states=[];
index=1;
for i=1:I    
    for j=1:J
                states(index,1)=x1(i);
                states(index,2)=x2(j);
                index=index+1;
                   
    end
end
