function [ s ] = getDisState( x, binIntervals, numBins )
%GETDIMSTATE Summary of this function goes here
%   Detailed explanation goes here
s = 0;
stepper = 1;
for d=1:size(binIntervals,2)
    theBin = 1;
    acc = binIntervals(d);
    while(acc <= x(d)) 
        acc = acc + binIntervals(d);
        theBin = theBin + 1;
    end
    s = s + stepper * (theBin-1);
    stepper = stepper * numBins(d);
end

s = s + 1;
end

