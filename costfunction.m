function [ jVal, gradient ] = costfunction(thetha,X,Y)
fprintf('Halo 0 \n')
%COSTFUNCTION Summary of this function goes here
%   Detailed explanation goes here
m=length(Y);
%fprintf('Halo 1 \n')
fprintf('Size m %d \n',m)

hthetha=sigmoid(X*thetha);
%fprintf('Halo 2 \n')
%fprintf('Size hthetha %d \n',size(hthetha))

A=Y.*log(hthetha);
%fprintf('Halo 3 \n')
B=(1-Y).*log(1-hthetha);
%fprintf('Halo 4 \n')
jVal=(-1/m)*sum(A+B);
jVal
%fprintf('Halo 5 \n')
gradient=(1/m).*(hthetha-Y)'*X;
%fprintf('Halo 6 \n')
end

