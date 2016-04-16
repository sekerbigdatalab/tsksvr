% predict fuzzy svm

% If you use our method, please cite our article:
% V. Uslan and H. Seker, “Quantitative prediction of peptide binding
% affinity by using hybrid fuzzy support vector regression,” 
% Applied Soft Computing, vol. 43, pp. 210–221, 2016.

function [out]=predictfsvm(X,D,M,sigma,C,bias)

[L,n]=size(X); % L is the number of data samples ==========================
[m,n]=size(M); % m is the number of rules =================================

out = [];

labels = D;
features = zeros(L,m*(n+1));

A = zeros(L,m*(n+1));
weights = zeros(L,m);

for i=1:L % L is the number of data samples ===============================
U=[]; % each data sample begins with empty U ==============================
for j=1:m % m is the number of rules ====================================== 
u=1; % each rule begins with empty u ======================================
for t=1:n % n is the number of input variables ============================
u=u*(gaussmf(X(i,t),[sigma(j,t),M(j,t)])); % apply product tnorm ==========
end
U=[U,u]; % weights are found for each rule of the data sample =============
end

fa=U/sum(U); % this is the weight

% f is the ytsk
% =========================================================================

fa=fa';

% compute the consequent ==================================================
c0=C(:,n+1);
for t=1:n
c0=c0+C(:,t)*X(i,t);
end
% =========================================================================

f=fa'*c0 + bias; % this is the weighted average
out = [out f];

end  % for i=1:L ...

out = out';









