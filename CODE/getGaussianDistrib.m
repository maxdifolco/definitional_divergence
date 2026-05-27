function [gDistrib,OUT] = getGaussianDistrib(coords,N)

    % coords: nDim x nPoints
    IN.I = coords;
    IN.numDimMax = size(coords,2)-1;
    OUT = PCA(IN);
    OUT.eigVals(OUT.eigVals < 0) = 0;
    
    r1 = randn(N,IN.numDimMax) .* repmat( sqrt(OUT.eigVals') , [N,1]);
    
    gDistrib = (OUT.eigVecs(:,1:IN.numDimMax) * r1')' + repmat(mean(coords'),[N,1]);

end
