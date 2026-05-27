function OUT = diffusionMaps(IN)


%% Algo Coifman_2006: Diffusion Maps 
% Input arguments:
%     - IN.data: nbDims x nbCases
%     - IN.kNN
%     - IN.t: diffusion time
%     - IN.alphaT: first normalization parameter / 1=anisotropic kernel (Laplace-Beltrami) / 0.5=Fokker-Planck / 0=normalized graph Laplacian
% Output arguments:
%     - OUT.eigVecs
%     - OUT.eigVals
%     - OUT.coords
%     - OUT.K: Normalized affinity matrix
%     - OUT.Kold: Non-normalized affinity matrix 
%     - OUT.sigmaUSED: kernel bandwidth

[Np,Ns] = size(IN.data);

%% Step #1

tmpKS = squareform(pdist(IN.data'));

if isfield(IN,'alphaT')
    alphaT = IN.alphaT;
else    
    alphaT = 0;
end

if isfield(IN,'sigma')
    sigma = IN.sigma;
else    
    tmp = tmpKS + diag( Inf(Ns,1) );
    tmpB = sort(tmp,1,'ascend');
    tmpB = tmpB(IN.kNN,:); %% approximate density based on the nearest neighbors
    sigma = 3*mean(tmpB(:));
    clear tmp tmpB;
end
K = exp(-tmpKS.^2 / (2*sigma.^2));
clear tmpI tmpKS;

%% Step #2

Kold = K;
if alphaT ~= 0
    p = (sum(K,1).^alphaT)';
    K = K ./ (p * p'); %% First normalization
end
p = sqrt(sum(K, 1))';
K = K ./ (p * p'); %% Second normalization = normalized graph Laplacian (proba)

K = K^(IN.t);

numDimMax = IN.numDimMax;

[eV,eE] = eig(K); %%% ORDERED FROM SMALLEST TO LARGEST
eV = fliplr(eV);
eE = flipud(diag(eE));
eV = eV(:,2:end);
eE = eE(2:end);

eV = eV(:,1:numDimMax);
eE = eE(1:numDimMax);

%% Step #3

OUT.K = K;
OUT.Kold = Kold;
OUT.eigVals = eE;
OUT.eigVecs = eV;

lambda_t = eE.^IN.t; 
lambda_t=ones(Ns,1)*lambda_t';

tmpC = zeros(Ns,Ns);
tmpC(1:Ns,1:numDimMax) = eV.*lambda_t;
OUT.coords = tmpC';

OUT.sigmaUSED = sigma;

end
