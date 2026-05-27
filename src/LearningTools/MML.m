function OUT = MML(IN)

%% Algo Valencia_2011: Multiple manifold learning

% Input arguments:
%     - IN.data: cell x nb descriptors (in cell: nbCases x nbDims)
%     - IN.mu: interactions weight
%     - IN.alphaT: first normalization parameter / 1=anisotropic kernel (Laplace-Beltrami) / 0.5=Fokker-Planck / 0=normalized graph Laplacian
%%%%%%%%%% option #1: re-use the results of previous diffusion maps
%     - IN.W: previous kernels
%%%%%%%%%% option #2: re-compute diffusion maps from the data
%     - IN.kNN: required for diffusion maps
%     - IN.numS: min size of the data
% Output arguments:
%     - OUT.eigVecs
%     - OUT.eigVals
%     - OUT.coords
%     - OUT.A: affinity matrix (normalized)
%     - OUT.W: affinity matrix for one descriptor (one cell for each)
%     - OUT.M: extradiagonal matrix

Nfeat = length(IN.data);
[Nk,Np] = size(IN.data{1});

if isfield(IN,'W')
    W = IN.W;
else    
    tmpIN = IN;
    for s=1:Nfeat
        X = IN.data{s};
        tmpIN.I = X';
        tmpIN.data = tmpIN.I;
        tmpOUT = diffusionMaps(tmpIN);
        W{s} = tmpOUT.K;    
    end
end

M = cell(Nfeat,Nfeat);
A0 = zeros(Nfeat*Nk,Nfeat*Nk);

A = zeros(Nfeat*Nk,Nfeat*Nk);
for s=1:Nfeat
    for q=1:Nfeat
        if q==s
            idx = (1:Nk) + Nk*(s-1);
            A(idx,idx) = W{s};
            M{s,s} = W{s};
            
            A0(idx,idx) = W{s};
        else
            if IN.HAM == 1
                M{s,q} = eye(Nk);
            else
                M{s,q} = ( W{s}' * W{q} ) ./ ( repmat( sum(W{s}.^2,1).^(1/2) , [Nk,1] )' .* repmat( sum(W{q}.^2,1).^(1/2) , [Nk,1] ) );
            end
            
            idx_s = (1:Nk) + Nk*(s-1);
            idx_q = (1:Nk) + Nk*(q-1);
            A(idx_s,idx_q) = IN.mu*M{s,q};
            A(idx_q,idx_s) = IN.mu*M{s,q}';
            
            A0(idx_s,idx_q) = M{s,q};
            A0(idx_q,idx_s) = M{s,q}';
        end
    end
end

Aold = A;

if IN.alphaT ~= 0
    p = (sum(A,1).^IN.alphaT)';
    A = A ./ (p * p'); %% First normalization
end
p = sqrt(sum(A, 1))';
A = A ./ (p * p'); %% Second normalization = normalized graph Laplacian (proba)

[eV,eE] = eig(A); %%% ORDERED FROM SMALLEST TO LARGEST
eV = fliplr(eV);
eE = flipud(diag(eE));
eV = eV(:,2:end);
eE = eE(2:end);

OUT.eigVals = eE;
OUT.eigVecs = eV;
OUT.A = A;
OUT.A0 = A0;
OUT.M = M;
OUT.W = W;

Ns_big = size(A,1);
lambda_t = ones(Ns_big,1)*eE';

tmpC = eV.*lambda_t;
tmpC = tmpC';

OUT.coords = cell(Nfeat,1);
for s=1:Nfeat
    idx = (1:Nk) + Nk*(s-1);
    tmp = tmpC(:,idx);
    OUT.coords{s} = tmp;
end

end
