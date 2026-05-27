function descIncertitude = uncertaintyNdesc(IN,idx_highestDiff,matName)

outMML = IN.outMML;

nbDesc = length(IN.data);

for k = 1:nbDesc
    C{k} = outMML.coords{k};
    D{k} = IN.data{k};
end

gammaList = 1; newIN.gammaList = gammaList;
nDim = 90; 
N = 100; % number of point generated randomly

descIncertitude = NaN(size(D{1},1),size(IN.data{k},2));
for ki = idx_highestDiff
    
    %% MML
    newIN.yi = D{1}([1:ki-1,ki+1:end],:); % By default the first descriptor 
    newIN.xi = C{1}(1:nDim,[1:ki-1,ki+1:end])';

    cKi = [];
    for k = 1:nbDesc
        cKi = [cKi; C{k}(1:nDim,ki)'];
    end
    
    % Get distrib
    [newIN.x,~] = getGaussianDistrib(cKi',N);
    newIN.x = [newIN.x; C{1}(1:nDim,ki)']; % At the end: reconstruction of the descriptor or the mean (not used here)
    OUT = interpolateMulti(newIN);
    avgKi = mean(OUT.OUT{1}(1:end-1,:)); % mean of the samples generated 

    descIncertitude(ki,:) = std( OUT.OUT{1}(1:end-1,:) , 0,1 );
    
end 

end
