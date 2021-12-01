function [X, outliers] = addOutliers(Iu, Ju, X, m, n, frac, mu, sigma, type, sign, sampling)
% Add outliers on matrix X with index array Iu and Ju (can be double or
% int32) of size m x n
% Outliers are added with probability 0 <= frac <= 1
% Outliers are the addition of one realisation of S * N(mu, sigma^2) where S
% is a random "sign" variable and N(mu, sigma^2) a normal variable of mean
% mu and variance sigma^2
% type can either be 'additive' or 'multiplicative'. If multiplicative,
% outliers are the multiplication of X with S * N(mu, sigma^2) instead of
% the addition
% sign can either be 'sign' or 'nosign'. nosign means S = 1
% sign can either be 'uniform', 'linear' or 'hard'. These last two are
% other sampling mechanism, but are not used anymore.
%
% Example :
% [Xout, nOut] = addOutliers(Iu, Ju, Xtrue, m, n, 0.05, 1, 1, 'additive',
% 'sign', 'uniform')

% Mass tests
% Types
I = double(Iu) ;
J = double(Ju) ;
if any(size(I) ~= size(X)) || any(size(J) ~= size(X))
    error('Wrong sizes');
end
if ~ isscalar(m) || ~ isscalar(n) || ~ isscalar(frac) || ~ isscalar(mu) || ~ isscalar(sigma)
    error('m, n, frac, mu and sigma must be scalars') ;
end
if ~ ischar(type) || ~ ischar(sign) || ~ ischar(sampling)
    error('type, sign and sampling must be string') ;
end
% Valeurs
if ~ (strcmp(type,'multiplicative') || strcmp(type, 'additive'))
    error('Type should be mult. or additive.') ;
end
if ~ (strcmp(sign,'sign') || str(sign,'nosign'))
    error('Sign should be sign or nosign') ;
end
if ~ (strcmp(sampling,'uniform') || strcmp(sampling,'linear') || strcmp(sampling,'hard'))
    error('Sampling should be uniform or linear') ;
end
% If mult and sign -> no sense
if strcmp(sampling,'multiplicative') && strcmp(sign,'sign')
    warning('Mult. and sign : doesn''t makes sense ...') ;
end

% Gogogo
fprintf('Adding outliers (%3.2f percent)...',100*frac) ;
outliers = 0 ;
multiplicative = strcmp(type,'multiplicative') ;
additive = strcmp(type,'additive') ;
signRandom = strcmp(sign,'sign') ;
signNotRandom = strcmp(sign,'nosign') ;
uniformSampling = strcmp(sampling,'uniform') ;
linearSampling = strcmp(sampling, 'linear') ;
hardSampling = strcmp(sampling,'hard') ;

for i = 1:numel(X)
    
    addOutlier = false ;
    
    randSampling = rand() ;
    if uniformSampling 
        if randSampling <= frac
            addOutlier = true ;
        end
    elseif linearSampling
        if randSampling <= frac * 2 * J(i) / n
            addOutlier = true ;
        end
    elseif hardSampling
        if n/4 <= J(i) && J(i) <= 3*n/4 && randSampling <= 2 * frac 
            addOutlier = true ;
        end
    else
        error('Sampling type wrong') ;
    end
    
    if addOutlier   
        
        N = (mu + sigma * randn) ;  
        
        if signRandom
            random = rand() ;
            S = (random > 0.5) * (-1) + (random <= 0.5) * 1 ;
        elseif signNotRandom
            S = 1 ;
        else
            error('Sign wrong value') ;
        end
        
        if multiplicative
            X(i) = X(i) * N ;
        elseif additive
            X(i) = X(i) + S * N ;
        else
            error('Type should be additive or multiplicative') ;
        end
        
        outliers = outliers + 1 ;  
        
    end
end

fprintf(' %3.2f percent (%d) added.\n',100*outliers/numel(X),outliers) ;

end