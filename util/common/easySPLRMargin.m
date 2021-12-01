function [U, V, stats] = easySPLRMargin(problem, params)

Omega = sparse(problem.I, problem.J, ones(size(problem.A)), problem.m, problem.n) ;
M     = sparse(problem.I, problem.J,           problem.A,   problem.m, problem.n) ;
lam_lasso = params.lam_lasso ;
lam_nuc   = params.lam_nuc ;
Maxiter   = params.maxiter ;
Tolerance = params.tol ;
maxtime   = params.timelimit ;
verbose   = params.verbose ;

[X_old,Z_old,times_acc_nest,obj_vals_acc_nest,rnk_soln,U,V,stats] = sp_lr_margin(M,[],[],Omega,lam_lasso,lam_nuc,Maxiter,Tolerance) ;

%rnk_soln
%times_acc_nest
%obj_vals_acc_nest

    function [X_old,Z_old,times_acc_nest,obj_vals_acc_nest,rnk_soln,U,V,stats]=sp_lr_margin(B,X_old,Z_old,A,lam_lasso,lam_nuc,Maxiter,Tolerance)
        % this is an efficient version of sparse + low-rank optimization with
        % marginalization over the sparse part.
        
        %% mini_{X,Z} 0.5 ||A.*(X +Z - B)||_F^2 + lam_nuc*||X||_* + lam_lasso*||Z||_1
        
        
        %% X_old, Z_old: initial estimates (can be empty)
        %% A: sparse binary matrix (showing missingness pattern as in matrix completion)
        %% B is data
        %% lam_lasso,lam_nuc : tuning params (see above for opt problem)
        %% Maxiter,Tolerance: self evident
        
        
        restart = 0;
        
        initTime = tic() ;
        
        [nrow,ncol]=size(B);
        if isempty(Z_old)
            Z_old=zeros(nrow,ncol);
        end
        
        if isempty(X_old)
            X_old=Z_old;
        end
        
        X_couple=X_old;
        
        obj_vals_acc_nest=zeros(Maxiter,1);
        times_acc_nest=obj_vals_acc_nest;
        tk=1;
        tt0=A.*B;
        tt=tt0;
        
        LARGE=1; OPT.minSingValue=lam_nuc;
        
        for iter = 1: Maxiter
            
            t=cputime;
            
            %%Z_couple =sign(A.*(B-X_couple)).*max(abs(A.*(B-X_couple)) - lam_lasso,0); grad_vec=(A.*(X_couple+Z_couple - B));
            
            tt=tt0 - A.*X_couple; grad_vec=-sign(tt).*min(abs(tt),lam_lasso);
            
            if LARGE~=1
                [a1,a2,a3]=svd(X_couple - grad_vec,'econ');
            else
                [a1,a2,a3]=lansvd(X_couple - grad_vec,problem.r,'T',OPT);
                %[a1,a2,a3]=lansvd(X_couple - grad_vec,100,'L',OPT);
            end
            
            sing_vals=max(diag(a2)-lam_nuc,0);
            X_new=a1*diag(sing_vals)*a3';
            
            tk1 = 1+sqrt(1+4*(tk^2));
            tk1 = tk1/2;
            X_couple=X_new + (tk-1)*(X_new - X_old)/tk1;
            
            tk=tk1;
            
            %if (mod(iter,100)==0);    tk=1;end
            
            times_acc_nest(iter)= cputime-t;
            
            Z_old=sign(A.*(B-X_new)).*max(abs(A.*(B-X_new)) - lam_lasso,0);
            obj_vals_acc_nest(iter)=  0.5*norm(A.*(X_new +Z_old - B),'fro')^2+ lam_nuc*sum(sing_vals)...
                + lam_lasso*sum(abs(Z_old(:)));
            
            %% Compute UV factorization
            idxNnz = find(sing_vals) ;
            U = a1(:,idxNnz) ;
            V = diag(sing_vals(idxNnz))*a3(:,idxNnz)' ;
            rmse = sqrt(sqfrobnormfactors(U, V, problem.Utrue, problem.Vtrue)/(problem.m * problem.n)) ;
            if verbose == 1
                fprintf('%d | Obj : %e, RMSE : %e\n', iter, obj_vals_acc_nest(iter), rmse) ;
            end
            
            stats(iter).Time = toc(initTime) ;
            stats(iter).RMSE = rmse ;
            
            
            tol=obj_vals_acc_nest(max(iter-1,1));
            
            %%tol
            if (restart)&(obj_vals_acc_nest(iter) > tol);
                tk=1; X_new=X_old;
            end
                        
            X_old=X_new;
            
            
            tol =abs(obj_vals_acc_nest(iter) - tol)/ (tol + 1e-12);
            
            
            %if (tol<Tolerance)&(iter>1)
            if rmse < Tolerance & iter > 1
                times_acc_nest=times_acc_nest(1:iter);
                obj_vals_acc_nest=obj_vals_acc_nest(1:iter);
                stats = stats(1:iter) ;
                break
            end
            
            if toc(initTime) > maxtime
                fprintf('SPRLMargin stopping after timelimit') ;
                break ;
            end                        
        end
        
        rnk_soln=nnz(sing_vals);
        
    end

end