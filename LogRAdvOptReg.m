%Advance Opt
format long
file=csvread('train.csv',1,0);
%size(file)
Yold=file(:,1);
Xold=file(:,2:end);
%size(Yold)
%size(Xold)
%adding offset to Xold
X=[Xold ones(size(Xold,1),1)];
%X(:,end)
%size(X)
Y=zeros(size(Yold,1),10);
%size(Y)

%filling the classification
for num=0:9
    for i=1:size(Yold,1)
        if(Yold(i)==num)
            Y(i,num+1)=1;
        end
    end
end

%m=size(Yold,1);
options = optimset('LargeScale','off','GradObj', 'on','MaxIter',10000);
%options=optimset('TolX',1e-6,'TolFun',1e-6,'GradObj','on','MaxIter',10);
%options = optimset('LargeScale','off','GradObj', 'on');
%options = optimoptions(@fminunc,'Algorithm','quasi-newton');
initialThetha=zeros(size(X,2),10);
%lamda=1000;

%learning
for learn=1:10
    fprintf('Learning %d \n',learn-1)
    fparam=@(t)costfunctionreg(t,X,Y(:,learn));
    fprintf('Cek %d \n',learn-1)
    [optThetha(:,learn),functionVal,exitFlag,output]=fminunc(fparam,initialThetha(:,learn),options);
end


%writing
filesubmit=csvread('test.csv',1,0);
Xsubmit=[filesubmit ones(size(filesubmit,1),1)];
Ysubmitfinal=100*ones(size(filesubmit,1),1);

hthetha_submit=sigmoid(Xsubmit*optThetha);

for ii=1:10
    for i=1:size(hthetha_submit,1)
        if hthetha_submit(i,ii)>=0.5
            Ysubmitfinal(i)=ii-1;
        end
    end
end

csvwrite('submitAdvReg.csv',Ysubmitfinal)