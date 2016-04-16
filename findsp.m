function [spout] = findsp(expout,prdout)

spout=corr(expout,prdout,'type','Spearman');