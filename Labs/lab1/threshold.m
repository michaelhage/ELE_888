function [thresholdVal] = threshold(D)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[M,N]=size(D); 

f=D(:,1);  % feature samples
la=D(:,N); % class labels

m11 = mean(f(la==1)); % mean of the class conditional density p(x/w1)
std11 = std(f(la==1));% Standard deviation of the class conditional density p(x/w1)

m12 = mean(f(la==2));% mean of the class conditional density p(x/w2)
std12= std(f(la==2));% Standard deviation of the class conditional density p(x/w2)

syms x;
eqn = (1./(sqrt(2.*pi).*std11)) .* exp(-((x-m11)./std11).^2./2) - ...
    (1./(sqrt(2*pi).*std12)) .* exp(-((x-m12)./std12).^2./2) == 0;

thresholdVal = solve(eqn,x);

end

