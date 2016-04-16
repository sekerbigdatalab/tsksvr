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

load mm4trninp.csv; load task4trnout.csv;
load mm4chkinp.csv; load task4chkout.txt;
trnin = mm4trninp; datout = task4trnout;
chkin = mm4chkinp; chkout = task4chkout;

%%  Settings for the optimal solution of this task

% selected features of the optimal solution
load sf4_141;
numInp=141;

% number of clusters determined for the optimal solution
cname = 'fcm'; 
cnum=2; 
 
% svr parameters determined for the optimal solution
ker = 'linear';
svrc = 2.30; 
svrp = 0.45;

%% STRUCTURE IDENTIFICATION OF FUZZY MODELLING

coeff = zeros(cnum,numInp+1);
trndat = [seldatin datout];

% cluster analysis (ca)
[rows, cols] = size(trndat);
[ctr, U, OF] = fcm(trndat,cnum);

%% ANTECEDENT PARAMETER IDENTIFICATION OF FUZZY MODELLING (fsysid)

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

svmout = predictfsvm(seldatin,datout,M,inpstd,C,bias); trnsp = findsp(datout,svmout);
svmout = predictfsvm(selchkin,chkout,M,inpstd,C,bias); chksp = findsp(chkout,svmout);

%% RESULTS

fprintf('trnsp=%.4f chsp=%.4f\n',trnsp,chksp);
