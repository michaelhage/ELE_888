function [posteriors_x,g_x]=MultivariateBDT(x,Training_Data, i)

D = Training_Data;

[M,N]=size(D);    
 
f=D(:,i);  % feature samples
la=D(:,N); % class labels

labels = unique(la); % Unique Labels

%% Prior Probabilities

Pr = zeros([length(labels) 1]);
for i = 1:length(labels)
    Pr(i) = sum(la(:) == labels(i));
end

%% Mean and Standard Deviation
mean = zeros([length(labels) length(i)]);
std = zeros([length(labels) length(i)]);

for i = 1:length(labels)
    for j = 1:length(i)
        mean(i,j) = mean(f(la==i,j));
        std(i,j) = std(f(la==i,j));
    end 
end

%% Covariance Matrix

for i = 1:length(labels)
    temp = num2cell(f(la==i,:),1);
    covariance(i) = cov(temp);
end

end
