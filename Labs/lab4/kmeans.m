function [mu_new, idx, error] = kmeans(X, c, set)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if nargin < 3
    set = 0; 
end

% common parameters
n = length(X);

% max iterations
iter_max = 100;

% initialize means
disp("Initial Values")
mu = 255 * rand(3, c)

% initialize variables
class = zeros(n,c);
mu_new = zeros(3,c);
error = zeros(iter_max,1);

if(set == 1)
    idx = zeros(n,iter_max);
end

for i = 1:iter_max
    
%     identifying the class the image belongs to
    for j = 1:c
        class(:,j) = sqrt( (X(:,1) - mu(1,j)).^2 + (X(:,2) - mu(2,j)).^2 + (X(:,3) - mu(3,j)).^2 );
    end
    
%     finiding index of min value

    if(set == 0)
        [~,idx] = min(class,[],2);
        for j = 1:c
            mu_new(:,j) = mean( X(idx(:) == j, :) );
        end
        
    else
        [~,idx(:,i)] = min(class,[],2);
        for j = 1:c
            mu_new(:,j) = mean( X(idx(:,i) == j, :) );
        end
    end
    
    for x = 1:n
        error(i) = error(i) + sqrt( (X(x,1) - mu(1,idx(x))).^2 + (X(x,2) - mu(2,idx(x))).^2 + (X(x,3) - mu(3,idx(x))).^2 );
    end
    
    if( sum(mu - mu_new, 'all') == 0 )
         
        break;
    end
     
     mu = mu_new;
end

disp("number of iterations: " + num2str(i))
error = error(1:i);

if(set == 1)
    idx = idx(:,1:i);
end

end

