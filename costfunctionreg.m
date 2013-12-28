function [ jVal, gradient ] = costfunctionreg(thetha,X,Y)
%fprintf('Halo 0 \n')
lamda=100;
%COSTFUNCTION Summary of this function goes here
%   Detailed explanation goes here
m=length(Y);
%fprintf('Halo 1 \n')
%fprintf('Size m %d \n',m)

hthetha=sigmoid(X*thetha);
%fprintf('Halo 2 \n')
%fprintf('Size hthetha %d \n',size(hthetha))

A=Y.*log(hthetha);
%fprintf('Halo 3 \n')
B=(1-Y).*log(1-hthetha);
%fprintf('Halo 4 \n')
jVal=(-1/m)*sum(A+B)+(lamda/(2*m))*sum(thetha(1:(end-1)).^2);
%jVal
%fprintf('Halo 5 \n')
gradient=zeros(size(X,2),1);
gradient(1:end-1)=((1/m).*(hthetha-Y)'*X(:,1:end-1))'+(lamda/m).*thetha(1:end-1);
gradient(end)=(1/m).*(hthetha-Y)'*X(:,end);
%fprintf('Halo 6 \n')
end

