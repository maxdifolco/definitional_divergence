function output_03 = MINEXACT_Bermanis_ACHA_2013_ALGO_03_clean( B_s , f , x_s , x_star , e_s , gamma , i_list , NYSTROM , usePINV )

[n,l] = size(B_s);

if( (n == l) && (l == length(i_list)) )
    EYE = eye(n,n);
else
    EYE = zeros(n,l);
    for j=1:length(i_list)
        EYE(i_list(j),j) = 1; %%% points for which we want exact matching
    end
end

Ktmp = ( B_s + 1/gamma * EYE ); %%% analytical solution

clear EYE;

%% step 3
% usePINV = 1;
if( usePINV == 1)
    %%% slower version for small number of attributes: 
    if( n == l)
        B_s_cross = inv( Ktmp );
    else
        B_s_cross = pinv( Ktmp );
    end
    nD = size(f,2);
% % %     c = zeros(l,nD); for i=1:nD; c(:,i) = B_s_cross * f(:,i); end;
    c = (f' * B_s_cross)';
else
    nD = size(f,2);
% % %     c = zeros(l,nD); for i=1:nD; c(:,i) = Ktmp \ f(:,i); end;
    c = (f' / Ktmp)';
end

clear Ktmp B_s_cross;

%% step 4
% % % f_s = zeros(n,nD); for i=1:nD; f_s(:,i) = B_s * c(:,i); end; 
f_s = (c' * B_s)';

%% step 5
p = size(x_star,1);

tmp2 = pdist2(x_star,x_s);
    if( NYSTROM == 0 )
        tmp2 = exp( -tmp2.^2 / e_s );
    else
        tmp2 = exp( -tmp2.^2 / (2*e_s.^2) );
    end
f_star_s = (c' * tmp2')';

% % % f_star_s = zeros(p,nD);
% % % for j=1:p
% % %     tmp = repmat(x_star(j,:) , l , 1);
% % %     tmp = (tmp - x_s).^2;
% % %     tmp = sum(tmp,2);
% % %     if( NYSTROM == 0 )
% % %         G_star_s = exp( -tmp / e_s );  %% (l_s x 1)
% % %     else
% % %         G_star_s = exp( -tmp / (2*e_s.^2) );  %% (l_s x 1)
% % %     end
% % % %% step 6
% % %     f_star_s(j,:) = G_star_s' * c;
% % % end

output_03.f_s = f_s;
output_03.f_star_s = f_star_s;

end

