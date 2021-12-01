function [U, V, stats] = easyAOPMC(Iu, Ju, X, m, n, r, lambda, Xtrue, Utrue, Vtrue, nOut, params)
% Runs the AOPMC algorithm (see in-code for parameters)

Iu = uint32(Iu) ;
Ju = uint32(Ju) ;

if any(size(Utrue) ~= [m r]) || any(size(Vtrue) ~= [r n]) || ~isscalar(nOut) || ...
    ~isscalar(m) || ~isscalar(n) || ~isscalar(r) || ~isscalar(lambda) || ...
    any(size(Iu) ~= size(Ju)) || any(size(Iu) ~= size(X)) || any(size(Iu) ~= size(Xtrue))
    error('Wrong sizes') ;
end

Known = reshape(sub2ind([m n], double(Iu), double(Ju)), [1 length(X)]) ;
data = reshape(X, [1 length(X)]) ;
L = nOut ;

maxiter = 20 ;
opts.maxouter = 20 ;
opts.maxinner = 20 ;
opts.tolgradnorm = 1e-8;
opts.verbosity = 2;
opts.order = 2;
opts.computeRMSE = false;
opts.computeMAE = true;
opts.U0 = [] ;
opts.Xtrue = Xtrue ;
opts.Xout = X ;
opts.I0u = Iu ;
opts.J0u = Ju ;
opts.timelimit = params.timelimit ;
opts.Utrue = Utrue ;
opts.Vtrue = Vtrue ;

[U, V, stats, ~] = BMP_rtrmc(m, n, r, lambda, Known, data, L, maxiter, opts) ;

end

% The following code was provided by the authors themselves, with some
% modifications done by me
function [X, Y, stats, Out] = BMP_rtrmc(m, n, r, lambda, Known, data, L, maxiter, opts)
%%
% BMP_rtrmc is the function used to do robust matrix completion. It detects 
% the locations of outliers and use rtrmc to do matrix completion using 
% only the 'correct' entries.
% 
%                Minimize 0.5* sum_{(i,j) in Known}Lambda_{ij}((X*Y)_{ij}-X_{ij})^2 
%			  s.t.  sum(1-Lambda_{ij}) <= L, Lambda_{ij} = 0 or 1.
%
% In this case L will only decrease when L is overestimated. For only some
% cases, it will return the correct L value. In order to find the correct L
% value, BMP_rtrmc_all, which utilize BMP_rtrmc for several times, has to be used.
%
%			
% Inputs are 
%            m:   number of rows                       
%            n:   number of columns
%            r:   rank of the matrix (m-by-n) to be recovered
%       lambda:   a very small paramter i.e., 1e-8
%        Known:   the positions of known entries
%         data:   the corresponding values of known entries
%            L:   the number of outliers
%      maxiter:   The maximum total number of iterations
%         opts:   including all parameters for rtrmc                       
%
% Outputs are
%            X:   matrix of size (m-by-r)  (X^T*X=I)
%            Y:   matrix of size (r-by-n)
%        Stats:   stats for plot
%          Out:   other outputs
%                                                                       
%
% Author:    Ming Yan (basca.yan@gmail.com) and Yi Yang (yyang@math.ucla.edu)
%   Date:    2012-2-24(UCLA)
%
% Reference:  M. Yan, Y. Yang and S. Osher, Exact low-rank matrix completion 
%             from sparsely corrupted entries via adaptive outlier pursuit. 
%             UCLA CAM report 12-34. 
%
%%
    startAlgo   = tic() ;
    iter        = 0;    
    Loc         = 1:length(Known); 
    [Ik, Jk]    = ind2sub([m n], Known(Loc));
    problem     = buildproblem(Ik, Jk, data(Loc), [], m, n, r, lambda);
    problem.A   = opts.Utrue ;
    problem.B   = opts.Vtrue ;
    if isempty(opts.U0)        
        U0 = initialguess(problem);
    else
        U0 = opts.U0;
    end
    error       = 0; % a counter used to terminate the algorithm when it reaches a given number (e.g. 5)
    initTime = toc(startAlgo) ;

    while iter < maxiter  
        fprintf('AOPMC - Iteration %d\n', iter) ;
        % rtrmc
        opts.maxtime = opts.timelimit - toc(startAlgo) ;
        [X, Y, stat] = rtrmc(problem, opts, U0);          
        
        % STATS
        % A few other things (leo)
        stats(iter+1).MAE = [stat.MAE] ;        
        stats(iter+1).RMSE = [stat.RMSE] ;
        stats(iter+1).Obj = [stat.cost] ;
        stats(iter+1).nOuts = [stat.nOut] ;
        if iter == 0
            stats(iter+1).Its = length(stat) ;
            stats(iter+1).Time = [stat.time] + initTime ; % Don t forget init time
        else
            stats(iter+1).Its = length(stat) + stats(iter).Its ;
            oldTime = stats(iter).Time ;
            stats(iter+1).Time = [stat.time] + oldTime(end) + toc(startAuxiliary) ;
        end
        stats(iter+1).nLambda = length(Ik) ;
        %
        if opts.maxtime < 0
            fprintf('AOPMC - Time limit reached, aborting') ;
            break ;
        end
        
        startAuxiliary = tic() ;
        % M_temp     = X*Y;                
        % y_t        = abs(M_temp(Known) - data);       
        M_temp = spmaskmult(X, Y, opts.I0u, opts.J0u) ;
        y_t = abs(M_temp' - data) ;
        % find the largest L elements of y_t
        if iter == 0
            [~, index] = sort(y_t, 'descend');
            index0     = index;
            % update Loc to be the rest length(Known) - L locations
            Loc        = 1:length(Known);
            Loc(index(1:L)) = [];
        else
            [~, index_new] = sort(y_t, 'descend');
            if length(union(index_new(1:L), index(1:L))) == L 
                if stat(end).cost < 1e-10 % used to be 1e-10
                    break;
                elseif error >= 5
                    break;
                else
                    if length(union(index_new(1:L), index0(1:L))) == L 
                        break;
                    else
                        opts.maxouter = 100;  % more iterations needed
                        opts.maxinner = 30;
                        error  = error + 1;
                        index0 = index_new;
                    end
                    
                end
            else
                opts.maxouter = 10;
                opts.maxinner = 3;
                Loc     = 1:length(Known);
                if y_t(index_new(L)) < 1e-5                % used to be 1e-7
                    L = sum(y_t(index_new(1:L)) > 1e-5); % update L if necessary
                end
                Loc(index_new(1:L)) = [];
                index = index_new;
            end
        end        
        [Ik, Jk] = ind2sub([m n], Known(Loc));
        problem  = buildproblem(Ik, Jk, data(Loc), [], m, n, r, lambda);
        problem.A   = opts.Utrue ;
        problem.B   = opts.Vtrue ;
        U0       = X;
        iter     = iter + 1;
        %checkMatrix = zeros(m,n);
        %checkMatrix(Known(Loc)) = 1;
        %checkVector = [sum(checkMatrix),sum(checkMatrix,2)'];
        %[min_orig,min_orig_loc] = min(checkVector);                
        
    end
    Out.Loc  = Loc;
    Out.stat = stat;
    Out.iter = iter;
    Out.L    = L;
    %Out.min_recover = min_orig;
    %Out.min_recover_loc = min_orig_loc;
end