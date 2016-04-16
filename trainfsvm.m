% train fuzzy svm

% If you use our method, please cite our article:
% V. Uslan and H. Seker, “Quantitative prediction of peptide binding
% affinity by using hybrid fuzzy support vector regression,” 
% Applied Soft Computing, vol. 43, pp. 210–221, 2016.

function [C,bias]=trainfsvm(X,D,M,sigma,ker,svrc,svrp)

[L,n]=size(X); % L is the number of data samples ==========================
[m,n]=size(M); % m is the number of rules =================================

trn_labels = D;
trn_features = zeros(L,m*(n+1));

trnA = zeros(L,m*(n+1));
weights = zeros(L,m);

itermax = 1;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for iter=1:itermax
    
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

xtemp = fa*[X(i,:) 1];
xtemp = reshape(xtemp',1,m*(n+1));
trnA(i,:) = xtemp;
weights(i,:) = fa';

end  % for i=1:L ...

trn_features = trnA;
trn_features_sparse = sparse(trn_features);
libsvmwrite('trn.dat',trn_labels,trn_features_sparse);

% Linear Kernel
[mg_trn_label, mg_trn_inst] = libsvmread('trn.dat');
param = ['-s 3 -t 0 -c ' num2str(svrc) ' -p ' num2str(svrp(1))];
model = libsvmtrain(mg_trn_label, mg_trn_inst, param);

w = model.SVs' * model.sv_coef;
b = -model.rho;

C = reshape(w,n+1,m)';
bias = b;

% compute the consequent ==================================================
c0=C(:,n+1);
for t=1:n
c0=c0+C(:,t)*X(i,t);
end
% =========================================================================

f=fa'*c0; % this is the weighted average

e=D(i)-f;

end % iter
