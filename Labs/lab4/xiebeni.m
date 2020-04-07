function [Y] = xiebeni(idx, X, mu, c)
% Question C

%     initialize variables
    mu_diff = zeros(c,1);
    Y = 0;
    
%     get difference from each individual mean cluster vector to its nearby
%     closest mean vector
    for i = 1:c
        mu_temp = mu;
        mu_temp(:,i) = [];
        
        mu_diff(i) = min( sqrt((mu(1,i) - mu_temp(1,:)).^2 + (mu(2,i) - mu_temp(2,:)).^2 + (mu(3,i) - mu_temp(3,:)).^2) );
    end
    
%     compute the Xie-Beni difference
    for i = 1:length(X)
        Y = Y + ( (X(i,1) - mu(1,idx(i)).').^2 + (X(i,2) - mu(2,idx(i)).').^2 + (X(i,3) - mu(3,idx(i)).').^2 ) ./ mu_diff(idx(i));
    end
    
%     get the average difference
    Y = Y / length(X);
end

