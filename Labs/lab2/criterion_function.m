function [a] = criterion_function(set_A,set_B, a, nu)
%CRITERION_FUNCTION Summary of this function goes here
%   Detailed explanation goes here

% weight factor parameters
if ~exist('nu','var')
    nu = 0.01;
end
if ~exist('a', 'var') || (a == 0)
    a = [0 0 1];
end
theta = 0;

% size of the sets
sizeA = size(set_A,1);
sizeB = size(set_B,1);

% Creation of the augmented array
augmented = ones(sizeA + sizeB, 1);
augmented(sizeA+1:end) = -1;

setB_aug = -set_B;

augmented = [augmented [set_A; setB_aug]];

aws(1,:) = a;
for iter = 1:300
%     disp(iter);
    
    classifier = aws(iter,:)*augmented.';
    
    i = find(classifier(:)<0);
    gradient = -sum(augmented(i,:),1);
    offset = nu .* gradient;
    
    aws(iter+1,:) = aws(iter,:) - offset;
    
    if abs(offset) < theta
        break;
    end
end
    a = aws(iter+1,:);
    disp("weight vector: " + num2str(a));
end