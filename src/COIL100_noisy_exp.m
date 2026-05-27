clear all;
close all;
clc;

folderRoot = '../data/';
addpath('./LearningTools');
addpath('./LearningTools/Regression');

R = [1,0,0];
G = [0,1,0];
colors = vertcat(R,G);

%% DATA READING

folderFiles = [folderRoot , '/COIL100/'];

testCase = 1; %%% 1 = toy bear / 2 = soda can

if testCase == 1
    num_obj = '52'; %%% toy bear
    list_viewT = [0:5:90,270:5:355]; %%% subset of views to prevent crossing loops in the latent space
    ilist=[1:7*11:407]; %%% subset of views to display
    dim2plot = [1,5,6]; %%% latent dimensions to plot / should display the data trajectory
else
    num_obj = '62'; %%% soda can
    list_viewT = (0:5:115); %%% subset of views to prevent crossing loops in the latent space
    ilist=[1:5*11:264]+5; %%% subset of views to display
    dim2plot = [1,2,3]; %%% latent dimensions to plot / should display the data trajectory
end
listInfo = [];

RED = [];
BLUE = [];
GREEN = [];

files = {};
for viewT = list_viewT

    filename = [folderFiles,'obj',num_obj, '__', num2str(viewT), '.png'];
    file = imread(filename);

    %%% augmenting the dataset with slight rotations +/- 5 degrees
    for angT=-5:1:5
        tmp = imrotate(file(:,:,1),angT,'bilinear','crop');
        RED = [RED ; reshape( squeeze(tmp),1,[] )];
        tmp = imrotate(file(:,:,2),angT,'bilinear','crop');
        GREEN = [GREEN ; reshape( squeeze(tmp),1,[] )];
        tmp = imrotate(file(:,:,3),angT,'bilinear','crop');
        BLUE = [BLUE ; reshape( squeeze(tmp),1,[] )];
        
        listInfo = [listInfo , [viewT;angT] ]; %%% for checking purposes: list of views + angles
    end
end

RED = double(RED);
GREEN = double(GREEN);
BLUE = double(BLUE);

close all;
clc;

%% VISUALIZING THE INPUT DATA

figure('units','normalized','position',[0,0,1,1],'color','w');
for ii=1:length(ilist) %%% displaying only RED and GREEN channels for this experiment
    i = ilist(ii);
    
    subplot(3,length(ilist),ii); %%% RGB image
    tmp = zeros(128,128,3);
    tmp(:,:,1) = reshape(RED(i,:),[128,128]);
    tmp(:,:,2) = reshape(GREEN(i,:),[128,128]);
    tmp(:,:,3) = reshape(BLUE(i,:),[128,128]);
    imagesc( uint8(tmp) );
    caxis([0 255]);
    axis square; axis off;
    title({['View angle: ',num2str(listInfo(1,i))],['Tilt angle: ',num2str(listInfo(2,i))]});
    
    subplot(3,length(ilist),ii+length(ilist)); %%% RED channel
    imagesc( reshape(RED(i,:),[128,128]) );
    axis square; axis off;
    colormap gray; caxis([0 255]);
    if ii==1
        title('RED CHANNEL');
    end
    
    subplot(3,length(ilist),ii+length(ilist)*2); %%% GREEN channel
    imagesc( reshape(GREEN(i,:),[128,128]) );
    axis square; axis off;
    colormap gray; caxis([0 255]);
    if ii==1
        title('GREEN CHANNEL');
    end
end

%% INDEPENDENT EMBEDDING WITH DIFFUSION MAPS / NO ALIGNMENT

IN.numS = size(RED,1); IN.numDimMax = size(RED,1)-1;
IN.alphaT = 1; IN.t = 1;
IN.kNN = 5;
IN.kNN2 = 5;
IN.mu = 1;

IN.I = RED'; IN.data = IN.I; IN.HAM = 0;
OUT_DM_R = diffusionMaps(IN);

IN.I = GREEN'; IN.data = IN.I; IN.HAM = 0;
OUT_DM_G = diffusionMaps(IN);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Affinity matrices on high- and low-dimensional samples
tmpK_R_HIGH = squareform(pdist(RED));
tmpK_G_HIGH = squareform(pdist(GREEN));
tmpK_R_LOW = squareform(pdist(OUT_DM_R.coords'));
tmpK_G_LOW = squareform(pdist(OUT_DM_G.coords'));

figure('units','normalized','position',[0,0,1,1],'color','w'); hold on;
subplot(2,2,1);
imagesc(tmpK_R_HIGH);
axis square; colorbar;
title('R channel: high-dimensional');
subplot(2,2,2);
imagesc(tmpK_G_HIGH);
axis square; colorbar;
title('G channel: high-dimensional');
subplot(2,2,3);
imagesc(tmpK_R_LOW);
axis square; colorbar;
title('R channel: low-dimensional');
subplot(2,2,4);
imagesc(tmpK_G_LOW);
axis square; colorbar;
title('G channel: low-dimensional');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tmp = {OUT_DM_R.coords , OUT_DM_G.coords};
figure('units','normalized','position',[0,0,1,1],'color','w'); hold on;
for u=1:2
	for k=1:IN.numS
        colorT = colors(u,:) * ( (k-1)/(IN.numS-1)*0.75 + 0.25 );
        plot3(tmp{u}(1,k),tmp{u}(2,k),tmp{u}(3,k),'.','MarkerSize',50,'Color',colorT);
        plot3(tmp{u}(1,k),tmp{u}(2,k),tmp{u}(3,k),'.','MarkerSize',50,'Color',colorT);
    end
end

axis square; grid on;
xlabel('dim1');
ylabel('dim2');
zlabel('dim3');
view([20,20]);

%% JOINT EMBEDDING WITH MML / ALIGNMENT + UNCERTAINTY ESTIMATION

numDimMax = IN.numS;


tic;
%%%%%%%% RED
IN.I = {RED, GREEN}; IN.data = IN.I;
IN.outMML = MML(IN);
outMML_RG = IN.outMML;
%%% finding the sample with highest uncertainty
tmp = (sum( ( IN.outMML.coords{1}(1:numDimMax,:) - IN.outMML.coords{2}(1:numDimMax,:) ).^2 , 1)).^(1/2);
idx_highestDiff_RG = find( tmp == min(tmp) ); %%% change to "min" for the lowest uncertainty
disp(listInfo(:,idx_highestDiff_RG));
%%% uncertainty on this specific sample
RG = divergenceNdesc(IN,idx_highestDiff_RG);
toc;

tic;
%%%%%%%% GREEN
IN.I = {GREEN, RED}; IN.data = IN.I;
IN.outMML = MML(IN);
outMML_GR = IN.outMML;
%%% finding the sample with highest uncertainty
tmp = (sum( ( IN.outMML.coords{1}(1:numDimMax,:) - IN.outMML.coords{2}(1:numDimMax,:) ).^2 , 1)).^(1/2);
idx_highestDiff_GR = find( tmp == min(tmp) ); %%% change to "min" for the lowest uncertainty
disp(listInfo(:,idx_highestDiff_GR));
%%% uncertainty on this specific sample
GR = divergenceNdesc(IN,idx_highestDiff_GR);
toc;

%% OUTPUT VISUALIZATION

list = {RG,GR};
title_list = {'RG','GR'};

list_MML = {outMML_RG,outMML_GR};

kID = [ idx_highestDiff_RG , idx_highestDiff_GR];

figure('units','normalized','position',[0,0,1,1],'color','w'); hold on;
for u=1:2 %%% RED and GREEN
    
    %%% images with the highest uncertainty
    subplot(2,2,u);
    imagesc(reshape(list{u}(kID(u),:),128,128)); 
    cmap = flipud(hot); colormap(cmap); caxis([0 50]);
    axis square; axis off;
    title(title_list{u});
    
    %%% aligned data
    subplot(2,2,u+2); hold on;
    if u == 1
        coords1 = list_MML{u}.coords{1};
        coords2 = list_MML{u}.coords{2};
    else
        coords1 = list_MML{u}.coords{2};
        coords2 = list_MML{u}.coords{1};
    end
    for k=1:IN.numS
        p = plot3([coords1(dim2plot(1),k),coords2(dim2plot(1),k)],[coords1(dim2plot(2),k),coords2(dim2plot(2),k)],[coords1(dim2plot(3),k),coords2(dim2plot(3),k)],'k','LineWidth',2.5);
        colorT1 = colors(1,:) * ( (k-1)/IN.numS*0.75 + 0.25 );
        p = plot3(coords1(dim2plot(1),k),coords1(dim2plot(2),k),coords1(dim2plot(3),k),'.','Color',colorT1,'MarkerSize',30);
        colorT2 = colors(2,:) * ( (k-1)/IN.numS*0.75 + 0.25 );
        p = plot3(coords2(dim2plot(1),k),coords2(dim2plot(2),k),coords2(dim2plot(3),k),'.','Color',colorT2,'MarkerSize',30);
        p.Color(4) = 1;
    end

    coords1 = list_MML{u}.coords{1};
    coords2 = list_MML{u}.coords{2};
    plot3(coords1(dim2plot(1),kID(u)),coords1(dim2plot(2),kID(u)),coords1(dim2plot(3),kID(u)),'x','Color',colors(1,:),'MarkerSize',40,'LineWidth',2);
    plot3(coords2(dim2plot(1),kID(u)),coords2(dim2plot(2),kID(u)),coords2(dim2plot(3),kID(u)),'x','Color',colors(2,:),'MarkerSize',40,'LineWidth',2);
    set(gca,'FontWeight','bold','FontSize',14);
    axis square; grid on;

    xlabel('dimension 1','FontWeight','bold');
    ylabel('dimension 2','FontWeight','bold');
end

