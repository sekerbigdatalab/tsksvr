%% Initial Settings

% If you'd like to use this script, 
% then you should change the script for your case.
% If you use our method, please cite our article:
% V. Uslan and H. Seker, “Quantitative prediction of peptide binding
% affinity by using hybrid fuzzy support vector regression,” 
% Applied Soft Computing, vol. 43, pp. 210–221, 2016.

clc; clear; close;
addpath('data','libsvm');

%% DATA (The input is min-max normalized)

load mm3trninp.csv; load task3trnout.csv;
load mm3chkinp.csv; load task3chkout.txt;
trnin = mm3trninp; datout = task3trnout;
chkin = mm3chkinp; chkout = task3chkout;

%%  Settings for the optimal solution of this task
% Task 3 with three rules and 172 features 
% yielded a q2 value of 0.310 with 
% linear svr parameters C = 1.45 and e = 0.90

% feature selection is omitted for simplicity
% task3 features are provided in task3features.txt 
features=load('task3features.txt'); % 172/5787
numInp = numel(features); % 172
seldatin=trnin(:,features); % selected training inputs
selchkin=chkin(:,features); % selected testing inputs

% number of clusters determined
cname = 'fcm'; % fuzzy c-means clustering
cnum = 3; % number of clusters (rules) is three

% svr parameters determined
ker = 'linear'; % svr linear kernel
svrc = 1.45; % svr C parameter
svrp = 0.90; % svr epsilon parameter

%% STRUCTURE IDENTIFICATION OF FUZZY MODELLING

coeff = zeros(cnum,numInp+1);
trndat = [seldatin datout];

% cluster analysis (ca)
[rows, cols] = size(trndat);
[ctr, U, OF] = fcm(trndat,cnum); 

%% ANTECEDENT PARAMETER IDENTIFICATION OF FUZZY MODELLING

% parameter identification: (mean)
ctrinp = []; 
for i=1:cnum, ctrinp = [ctrinp ctr(i,1:numInp)]; end
M = reshape(ctrinp,numInp,cnum)'; % Antecedent Mean (M)

% parameter identification: (stddev)
inpstd = zeros(cnum,numInp); %stddev
for i=1:cnum
    
   u = U(i,:); % u is the membership values of data samples to the ith cluster
   v = ctr(i,1:numInp); % v is the mean values of the ith cluster
   
   n = size(seldatin,1); % n is the number of data
   numInp = size(seldatin,2); % number of inputs
   diff = zeros(n,numInp);

   suu = 0; % sum of u

   for j=1:n
       diff(j,:)=seldatin(j,:)-v;
   end

   for j=1:n
       suu = suu + u(j)^1;
   end

   val = ((diff.^2)'*(u.^1)')./suu; % val is the variable for variance
   val = val';
   
   inpstd(i,:) = sqrt(val); % Antecedent S
end
C = coeff; % Consequent C

%% CONSEQUENT PARAMETER IDENTIFICATION OF FUZZY MODELLING

[C,bias]=trainfsvm(seldatin,datout,M,inpstd,ker,svrc,svrp);

svmout = predictfsvm(seldatin,datout,M,inpstd,C,bias); trnq2 = findq2(datout,svmout);
svmout = predictfsvm(selchkin,chkout,M,inpstd,C,bias); chkq2 = findq2(chkout,svmout);

%% RESULTS

fprintf('trnq2=%.4f chkq2=%.4f\n',trnq2,chkq2);
delete('trn.dat');