function OUT = computePCA(IN)

[Nf,Ns] = size(IN.I);

if isfield(IN,'useTurk')
    useTurk = IN.useTurk;
else
    useTurk = 0;
end

if isfield(IN,'numDimMax')
    numDimMax = IN.numDimMax;
else
    numDimMax = min(Nf,Ns);
end

MEAN = mean(IN.I,2);

if useTurk ~= 1
    %%% Big computation
    COV = cov(IN.I');
    [V,D] = eig(COV); %%% COV is SYMMETRIC POSITIVE DEFINITE, eigs > 0
    V = fliplr(V);
    D = flipud(diag(D));
    OUT.eigVecs = V;
    OUT.eigVals = D;
else
    %%% Turk's faster variant
    A = (IN.I - repmat(MEAN,[1,Ns])) / sqrt(Ns-1);
    iCOV = ( A' * A ); %%% beware, this is not cov(IN.I)
    [V,D] = eig(iCOV);
    V = fliplr(V);
    D = flipud(diag(D));
    VT = A * V;
    VT = VT ./ repmat( ( sum( VT.^2 , 1 ) ).^(1/2) , [Nf,1]); %%% This was somehow commented, I don't remember why
    OUT.eigVecs = VT;
    OUT.eigVals = D;
end

OUT.mean = MEAN;

if(Ns < Nf)
    OUT.eigVals = OUT.eigVals(1:Ns);
    OUT.eigVecs = OUT.eigVecs(:,1:Ns);
end

OUT.coords = OUT.eigVecs' * (IN.I - repmat(OUT.mean,[1,Ns]));

OUT.eigVals = OUT.eigVals(1:numDimMax);
OUT.eigVecs = OUT.eigVecs(:,1:numDimMax);
OUT.coords = OUT.coords(1:numDimMax,:);

end
