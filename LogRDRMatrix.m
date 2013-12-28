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

%start learning
%start finding the cost function (using linear model)
%Ysubmit=zeros(size(Yold));

eps=1e-6;
alpha=0.000001;
m=size(Yold,1);

%Learning for number 0
thetha=zeros(size(X,2),10); %saya ubah2 disini, pertama mulai dari 0, yang sekarang mulai dari 1
%mencari thetha aktuil

for i=1:10
    beta=ones(size(Yold));
    hthetha=ones(size(Yold));
    cost=1e6;
    diff=10;
    while diff>=eps
        costupdate=cost;
        %update the cost
        beta=X*thetha(:,i);
        denum_hthetha=1+exp(-beta);
        hthetha=1./denum_hthetha;
        A=Y(:,i).*log(hthetha);
        B=(1-Y(:,i)).*log(1-hthetha);
        cost=(-1/m)*sum(A+B);
        
        %update theta
        
        minusizer=(1/m).*(hthetha-Y(:,i))'*X;
        thetha(:,i)=thetha(:,i)-alpha.*minusizer';
               
        %calculating loop
        diff=costupdate-cost;
        diff
    end
end

beta_cek=X*thetha;
denum_hthetha_cek=1+exp(-beta_cek);
hthetha_cek=1./denum_hthetha_cek;

Ysubmit=100*ones(size(Y));

for ii=1:10
    for i=1:size(hthetha,1)
        if hthetha_cek(i,ii)>=0.5
            Ysubmit(i)=ii-1;
        end
    end
end

filesubmit=csvread('test.csv',1,0);
Xsubmit=[filesubmit ones(size(filesubmit,1),1)];
Ysubmitfinal=100*ones(size(filesubmit,1),1);

beta_submit=Xsubmit*thetha;
denum_hthetha_submit=1+exp(-beta_submit);
hthetha_submit=1./denum_hthetha_submit;

for ii=1:10
    for i=1:size(hthetha_submit,1)
        if hthetha_submit(i,ii)>=0.5
            Ysubmitfinal(i)=ii-1;
        end
    end
end

csvwrite('submitAdv.csv',Ysubmitfinal)