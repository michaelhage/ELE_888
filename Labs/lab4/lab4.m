%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LAB 4, Unsupervised Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Image Properties
clc
clear

% load image and display
I = imread('house.tiff');
figure 
title("Original Image")
imshow(I);

% reshape image
dim = size(I);
X = reshape(I, dim(1) * dim(2), dim(3));
X = double(X);

% plot colorspace
figure
plot3(X(:,1), X(:,2), X(:,3),'.', 'Color','b');
xlabel("Red Channel")
ylabel("Green Channel")
zlabel("Blue Channel")

%% Question A
clc

% initialize number of classes
c = 2;

% use algorithm
[mu, idx, error] = kmeans(X, c, 1);

% select two iterations
n = size(idx,2);
iter = ceil((n-1) * rand(1,2));

% earliest iteration portrayed
figure
plot3(X(idx(:,min(iter)) == 1,1), X(idx(:,min(iter)) == 1,2), X(idx(:,min(iter)) == 1,3),'.', 'Color','b');
hold on
plot3(X( idx(:,min(iter)) == 2 ,1), X(idx(:,min(iter)) == 2,2), X(idx(:,min(iter)) == 2,3),'.', 'Color','g')
hold on
plot3(mu(1,:), mu(2,:), mu(3,:), '*', 'Color', 'r')
xlabel("Red Channel")
ylabel("Green Channel")
zlabel("Blue Channel")
title("Iteration " + min(iter))

% middle iteration portrayed
figure
plot3(X(idx(:,max(iter)) == 1,1), X(idx(:,max(iter)) == 1,2), X(idx(:,max(iter)) == 1,3),'.', 'Color','b');
hold on
plot3(X( idx(:,max(iter)) == 2 ,1), X(idx(:,max(iter)) == 2,2), X(idx(:,max(iter)) == 2,3),'.', 'Color','g')
hold on
plot3(mu(1,:), mu(2,:), mu(3,:), '*', 'Color', 'r')
xlabel("Red Channel")
ylabel("Green Channel")
zlabel("Blue Channel")
title("Iteration " + max(iter))

% final iteration portrayed
figure
plot3(X(idx(:,n) == 1,1), X(idx(:,n) == 1,2), X(idx(:,n) == 1,3),'.', 'Color','b');
hold on
plot3(X( idx(:,n) == 2 ,1), X(idx(:,n) == 2,2), X(idx(:,n) == 2,3),'.', 'Color','g')
hold on
plot3(mu(1,:), mu(2,:), mu(3,:), '*', 'Color', 'r')
xlabel("Red Channel")
ylabel("Green Channel")
zlabel("Blue Channel")
title("Final Iteration - Iteration " + n)

% Question C
% compute Xie-Beni algorithm
Y = xiebeni(idx(:,n), X, mu, c);
disp("The Xie-Beni Difference for c=2 is: " + Y);

% convert to unsigned 8-bit integer
mu = uint8(mu);

% reshape index vector to index matrix
[x,y,z] = size(I);
M = reshape(idx(:,n),x,y);

% create new image
img = uint8(zeros(size(I)));

% map each pixel to their domain counterpart
for i = 1:x
    for j = 1:y
        img(i,j,:) = mu(:,M(i,j)).';
    end
end

figure, imshow(img);

% plot error
figure
plot(error, '.-')
title("Error Across Iterations")
xlabel("Iterations")
ylabel("Error")
%% Question B
clc

% use kmeans algorithm
c = 5;
[mu, idx, error] = kmeans(X, c);

% plot domain RGB plot
color = ['b'; 'g'; 'm'; 'y'; 'k'];
figure
for i = 1:c
    plot3(X(idx(:) == i,1), X(idx(:) == i,2), X(idx(:) == i,3),'.', 'Color',color(i));
    hold on
end
plot3(mu(1,:), mu(2,:), mu(3,:), '*', 'Color', 'r')
xlabel("Red Channel")
ylabel("Green Channel")
zlabel("Blue Channel")

% Question C
% Compute Xie-Beni algorithm
Y = xiebeni(idx, X, mu, c);

disp("The Xie-Beni Difference for c=5 is: " + Y);

% create domain image
mu = uint8(mu);
[x,y,z] = size(I);
M = reshape(idx(:),x,y);
img = uint8(zeros(size(I)));

% map each pixel to its colour space
for i = 1:x
    for j = 1:y
        img(i,j,:) = mu(:,M(i,j)).';
    end
end

figure, imshow(img);

% plot error
figure
plot(error, '.-')
title("Error Across Iterations")
xlabel("Iterations")
ylabel("Error")
