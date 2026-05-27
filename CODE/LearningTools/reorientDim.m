function feature2OUT = reorientDim(feature1,feature2,Ndims,normalizeT)

tmp1 = feature1; %%% dims x samples

if (~exist('normalizeT','var')) || sum(normalizeT==[0,1])==0
    normalizeT = 0;
end

%%% Permutation feature2 vs. feature1
tmpC = zeros(2^Ndims,1);
tmp2 = cell(2^Ndims,1);
for i=1:2^Ndims
    comb = dec2bin(i-1,Ndims); %%% string
    individualBits = comb - '0'; %%% breaks into individual bits
    tmp2{i} = feature2; %%% dims x samples
    tmp2{i}(individualBits==1,:) = -tmp2{i}(individualBits==1,:);
    
    %%% normalize variance
    tmp1n = tmp1(1:Ndims,:);
    tmp2n = tmp2{i}(1:Ndims,:);
    if normalizeT
        for d=1:Ndims
            tmp1n(d,:) = tmp1n(d,:) / std(tmp1n(d,:),0,2);
            tmp2n(d,:) = tmp2n(d,:) / std(tmp2n(d,:),0,2);
        end
    end
    
    tmpD = (tmp2n - tmp1n).^2;
    tmpC(i) = sum(tmpD(:));
end
idx = find(tmpC==min(tmpC));
feature2OUT = tmp2{idx};

end
