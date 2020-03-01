function plot_function(setA,setB, a)
%PLOT_FUNCTION Summary of this function goes here
%   Detailed explanation goes here
    x1 = 4;
    x2 = 7;
    y1 = 1;
    y2 = 5;
    
    x = linspace(x1,x2,100);
    y = abs(a(1)./a(3)) + abs(a(2)./a(3)).*x;
    
    figure

    plot(setA(:,1),setA(:,2),'rs'); title('x_1 vs x_2');
    hold on;
    plot(setB(:,1),setB(:,2),'k.');
    hold on
    plot(x,y);
    axis([x1 x2 y1 y2]);
end

