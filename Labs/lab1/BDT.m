function [posteriors_x,g_x]=BDT(x,Training_Data, i)

% x = individual sample to be tested (to identify its probable class label)
% featureOfInterest = index of relevant feature (column) in Training_Data 
% Train_Data = Matrix containing the training samples and numeric class labels
% posterior_x  = Posterior probabilities
% g_x = value of the discriminant function

D=Training_Data;

% D is MxN (M samples, N columns = N-1 features + 1 label)
[M,N]=size(D);    
 
f=D(:,i);  % feature samples
la=D(:,N); % class labels


%% %%%%Prior Probabilities%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hint: use the commands "find" and "length"

disp('Prior probabilities:');
Pr1 = sum(la(:) == 1) / M;
Pr2 = sum(la(:) == 2) / M;

%% %%%%%Class-conditional probabilities%%%%%%%%%%%%%%%%%%%%%%%

disp('Mean & Std for class 1 & 2');
m11 = mean(f(la==1)); % mean of the class conditional density p(x/w1)
std11 = std(f(la==1));% Standard deviation of the class conditional density p(x/w1)

m12 = mean(f(la==2));% mean of the class conditional density p(x/w2)
std12= std(f(la==2));% Standard deviation of the class conditional density p(x/w2)

% disp(['Conditional probabilities for x=' num2str(x)]);

% use the above mean, std and the test feature to calculate p(x/w1)
pxw1= (1./(sqrt(2.*pi).*std11)) .* exp(-((x-m11)./std11).^2./2);

% use the above mean, std and the test feature to calculate p(x/w2)
pxw2= (1./(sqrt(2*pi).*std12)) .* exp(-((x-m12)./std12).^2./2);

%% %%%%%%Compute the posterior probabilities%%%%%%%%%%%%%%%%%%%%

disp('Posterior prob. for the test feature');

% p(w1/x) for the given test feature value
pos11= pxw1*Pr1;

% p(w2/x) for the given test feature value
pos12= pxw2*Pr2;

posteriors_x= [pos11 pos12];

%% %%%%%%Discriminant function for min error rate classifier%%%

disp('Discriminant function for the test feature');

g_x= pos11-pos12;% compute the g(x) for min err rate classifier.