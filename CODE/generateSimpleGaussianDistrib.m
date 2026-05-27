clear all;
close all;
clc;

dimCase = [2,3];

N = 100;
t = linspace(0,2*pi,100);
t2 = linspace(-3,3,5);

figure;

for d=dimCase
    switch d
        case 2
            z1 = [1,1];
            z2 = [5,2];
            coords = [z1;z2]'; %%% nDim x nPoints
        case 3
            z1 = [0,1];
            z2 = [3,5];
            z3 = [-1,3];
            coords = [z1;z2;z3]'; %%% nDim x nPoints
    end
        
    [gDistrib,OUT] = getGaussianDistrib(coords,N);
    
    subplot(1,2,d-1); hold on;
    for i=1:size(gDistrib,1)
        plot(gDistrib(i,1),gDistrib(i,2),'r+');
    end
    for i=1:size(coords,2)
        plot(coords(1,i),coords(2,i),'b.','MarkerSize',40);
    end
    grid on;
    axis equal;

    
    if d==2
        tmp = (z2 - z1)';
        ax1 = tmp / norm(tmp);
        ax2 = [0,-1;1,0] * ax1;
        P = [ax1,ax2];
        D = [sqrt(OUT.eigVals),0];
        axis([-6,12,-2,4]);
    else
        P = OUT.eigVecs;
        D = sqrt(OUT.eigVals);
        axis([-6,7,-3,10]);
    end

    x0 = D(1) * cos(t);
    y0 = D(2) * sin(t);
    z = [x0;y0];
    zRot = P * z + mean(coords,2);
    plot(zRot(1,:),zRot(2,:),'k');
    for dim=1:(d-1)
        z = zeros(length(t2),2);
        z(:,dim) = sqrt(OUT.eigVals(dim)) * t2;
        zRot = P * z' + mean(coords,2);
        plot(zRot(1,:),zRot(2,:),'k','Marker','.','MarkerSize',20);
    end

end
