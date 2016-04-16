% q2 = findq2(expout,prdout)

function [q2] = findq2(expout,prdout)

nSamples = numel(expout);

sum1 = 0;
for i=1:nSamples
    c = expout(i)-prdout(i);
    d = c^2;
    sum1 = sum1 + d;
end

sum2 = 0;
e = mean(expout);
for i=1:nSamples
    f = expout(i)-e;
    g = f^2;
    sum2 = sum2 + g;
end

q2 = 1-(sum1/sum2);

%           SUM (exp - prd)^2
% q^2 = 1 - -----------------
%           SUM (exp - avg)^2