%function compare_spca
function info = demo_compare_SPCA_setting1()
clear;
close all;
addpath ../misc
addpath ../SSN_subproblem
%n_set=[ 200; 300; 500; ]; %dimension 
n_set=[ 200; 300; 400; 500; ]; %dimension 
%n_set = 500;
%format long
r_set = [5;10;15];   % rank
%r_set = [5;8;10;12;15];   % rank
%r_set = 5;

mu_set = [0.5;0.6;0.8];
%mu_set = [0.5;0.6;0.7;0.8];
%mu_set = 0.5;

%% problem setting
problem.cost_f = @cost_f;
    function f = cost_f(X,BX)
        if ~exist('BX', 'var')    
            f = -sum(sum(AtA*X.*X));
        else
            f = -sum(sum(BX.*X));
        end
    end

problem.cost_g = @cost_g;
    function f = cost_g(X)
        f = lambda*sum(sum(abs(X)));
    end


problem.prox_g = @prox_g;
    function y = prox_g(X,mu)
        y = max(abs(X) - mu*lambda*1,0).* sign(X);
    end

problem.egrad = @egrad;
    function g = egrad(X,BX)
        if ~exist('BX', 'var')    
            g = -2*(AtA*X)/1;
        else
            g = -2*BX;
        end    
    end

problem.ehess = @ehess;
    function g = ehess(X,U)
        g = -(AtA*U);
    end

info.n = 500; info.r = 5; info.C = zeros(8,7,length(n_set));
%% cycle
for id_n = 1:size(n_set,1)        % n  dimension
    n = n_set(id_n);
    fid =1;    
    for id_r =2%:size(r_set,1) % r  number of column
        for id_mu = 2%:size(mu_set,1)         % mu  sparse parameter
            r = r_set(id_r);
            lambda = mu_set(id_mu);
            succ_no_manpg = 0;    succ_no_mialm = 0;     succ_no_admm = 0; 
            succ_no_manpg_BB = 0; succ_no_SOC = 0;  succ_no_PAMAL = 0; succ_no_sub = 0;
            diff_no_SOC = 0;  diff_no_PAMAL = 0;  diff_no_sub = 0;
            fail_no_SOC = 0;  fail_no_PAMAL = 0;  fail_no_sub = 0;
            residual_manpg = zeros(50,1);   residual_mialm = zeros(50,1);  residual_manpg_BB = zeros(50,1);  residual_admm = zeros(50,1);  
            residual_Rsub = zeros(50,1);    residual_PAMAL = zeros(50,1);  residual_soc = zeros(50,1);  
            A = zeros(8,7);
            for test_random = 1:20  %times average.
                fprintf(fid,'==============================================================================================\n');
                
                rng('shuffle');
                m = 50;
                B = randn(m,n);
                type = 0; % random data matrix
                if (type == 1) %covariance matrix
                    scale = max( diag(B)); % Sigma=A/scale;
                elseif (type == 0) %data matrix
                    B = B - repmat(mean(B,1),m,1);
                    B = normc(B);
                end
                
                AtA = B'*B;
                B = AtA; 
                type = 1;
                problem.M = stiefelfactory(n, r);
                problem.AtA = AtA;
                problem.B = 1;
                fprintf(fid,'- n -- r -- mu --------\n');
                fprintf(fid,'%4d %3d %3.3f \n',n,r,lambda);
                fprintf(fid,'----------------------------------------------------------------------------------\n');
                
                rng('shuffle');
               
                [phi_init,~] = svd(randn(n,r),0);  % random intialization
                
                option_Rsub.F_manpg = -1e10;
                option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 5e2;  option_Rsub.tol = 5e-3;
                option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=lambda;  option_Rsub.type = type;
                
                [phi_init, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
                    maxit_att_Rsub(test_random),succ_flag_sub]= Re_sub_grad_spca(B,option_Rsub);
                
                
                Init = phi_init;   %options_mialm.opt = F_manpg(test_random);
                options_mialm.stepsize = 1/(2*abs(eigs(full(AtA),1)));
                options_mialm.iter = 5000;    options_mialm.verbosity = 0;
                options_mialm.maxiter = 100; options_mialm.epso = 1e-8*n*r;
                options_mialm.tau = 0.99;    options_mialm.rho = 1.05;
                options_mialm.k = r;    %options_mialm.mu = 25/(n*r);
                options_mialm.mu = 0.5/svds(AtA,1)^1 ;
                options_mialm.tolgradnorm = 1; options_mialm.decrease = 0.9;
                options_mialm.AtA = AtA;       options_mialm.beta_type = 'P-R';

                 options_mialm.maxiter = 10;    
            
                
                 

                [X_mialm,F_mialm(test_random),sparsity_mialm(test_random),time_mialm(test_random),...
                    maxit_att_mialm(test_random),lins_mialm(test_random),in_av_mialm(test_random),succ_flag_mialm] = Riemannian_mialm_spca(problem, Init,options_mialm);

                if succ_flag_mialm == 1
                    succ_no_mialm = succ_no_mialm + 1;
                    %residual_mialm(test_random) = norm(u*u' - X_mialm*X_mialm','fro')^2;  
                end
                
                
                
                
               
                
                
                
                %%%%%  manpg parameter
                option_manpg.opt = F_mialm(test_random);
                option_manpg.adap = 0;    option_manpg.type =type;
                option_manpg.phi_init = phi_init; option_manpg.maxiter = 20000;  option_manpg.tol =1e-8*n*r;
                option_manpg.r = r;    option_manpg.n = n;  option_manpg.mu = lambda;
                %option_manpg.L = L;
                %option_manpg.inner_tol =1e-11;
                option_manpg.inner_iter = 100;
                %%%%%% soc parameter
                option_soc.phi_init = phi_init; option_soc.maxiter = 20000;  option_soc.tol =1e-4;
                option_soc.r = r;    option_soc.n = n;  option_soc.mu=lambda;
                %option_soc.L= L;
                option_soc.type = type;
                %%%%%% PAMAL parameter
                option_PAMAL.phi_init = phi_init; option_PAMAL.maxiter =20000;  option_PAMAL.tol =1e-4;
                %option_PAMAL.L = L;   option_PAMAL.V = V;
                option_PAMAL.r = r;   option_PAMAL.n = n;  option_PAMAL.mu=lambda;   option_PAMAL.type = type;
                %    B = randn(d,d)+eye(d,d); B = -B'*B;
                [X_manpg, F_manpg(test_random),sparsity_manpg(test_random),time_manpg(test_random),...
                    maxit_att_manpg(test_random),succ_flag_manpg, lins_manpg(test_random),in_av_manpg(test_random)]= manpg_orth_sparse(B,option_manpg);
                if succ_flag_manpg == 1
                    succ_no_manpg = succ_no_manpg + 1;
                    %residual_manpg(test_random) = norm(u*u' - X_manpg*X_manpg','fro')^2; 
                end
                

                option_manpg.F_manpg = F_mialm(test_random);
                [X_manpg_BB, F_manpg_BB(test_random),sparsity_manpg_BB(test_random),time_manpg_BB(test_random),...
                    maxit_att_manpg_BB(test_random),succ_flag_manpg_BB,lins_adap_manpg(test_random),in_av_adap_manpg(test_random)]= manpg_orth_sparse_adap(B,option_manpg);
                if succ_flag_manpg_BB == 1
                    succ_no_manpg_BB = succ_no_manpg_BB + 1;
                    %residual_manpg_BB(test_random) = norm(u*u' - X_manpg_BB*X_manpg_BB','fro')^2; 
               elseif(succ_flag_mialm == 1)
                    time_mialm(test_random) = 0;
                    F_mialm(test_random) = 0;
                    sparsity_mialm(test_random) = 0;
                    maxit_att_mialm(test_random) = 0;
                    lins_mialm(test_random) = 0;
                    in_av_mialm(test_random) = 0;
                    succ_no_mialm = succ_no_mialm  - 1;
                end
                
                
                
                
                
                
                
                %%%%%% Riemannian subgradient parameter
                option_Rsub.F_manpg = F_manpg(test_random);
                option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 1e3;      option_Rsub.tol = 5e-3;
                option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=lambda;  option_Rsub.type = type;
                
                [X_Rsub, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
                    maxit_att_Rsub(test_random),succ_flag_sub]= Re_sub_grad_spca(B,option_Rsub);
                %phi_init = X_Rsub;
                if succ_flag_sub == 1
                    succ_no_sub = succ_no_sub + 1;
                    %residual_Rsub(test_random) = norm(u*u' - X_Rsub*X_Rsub','fro')^2; 
                end
                option_soc.F_mialm = F_mialm(test_random);
                option_soc.X_mialm = X_mialm;
                option_PAMAL.F_mialm = F_mialm(test_random);
                option_PAMAL.X_mialm = X_mialm;
                [X_Soc, F_soc(test_random),sparsity_soc(test_random),time_soc(test_random),...
                    soc_error_XPQ(test_random),maxit_att_soc(test_random),succ_flag_SOC]= soc_spca(B,option_soc);
               % succ_flag_SOC = 1;
                if succ_flag_SOC == 1
                    succ_no_SOC = succ_no_SOC + 1;
                    %residual_soc(test_random) = norm(u*u' - X_Soc*X_Soc','fro')^2; 
                end
                %option_PAMAL.F_manpg = F_mialm(test_random);
                [X_pamal, F_pamal(test_random),sparsity_pamal(test_random),time_pamal(test_random),...
                    pam_error_XPQ(test_random), maxit_att_pamal(test_random),succ_flag_PAMAL]= PAMAL_spca(B,option_PAMAL);
               % succ_flag_PAMAL = 1;
                if succ_flag_PAMAL ==1
                    succ_no_PAMAL = succ_no_PAMAL + 1;
                    %residual_PAMAL(test_random) = norm(u*u' - X_pamal*X_pamal','fro')^2; 
                end
                
                
                
                %opt = min(F_mialm,F_manpg);
                
                
                
                
                options_admm = options_mialm;
                options_admm.mu = 0.5/svds(AtA,1)^1;
                options_admm.iter = 5000;  options_admm.opt = F_mialm(test_random);
                options_admm.maxiter = 100;
                options_admm.tolgradnorm = 1e-4;
                
                [X_admm,F_admm(test_random),sparsity_admm(test_random),time_admm(test_random),...
                    maxit_att_admm(test_random),lins_admm(test_random),in_av_admm(test_random),succ_flag_admm] = Riemannian_admm_spca(problem, Init, options_admm);
                if succ_flag_admm == 1
                    succ_no_admm = succ_no_admm + 1;
                   % residual_admm(test_random) = norm(u*u' - X_admm*X_admm','fro')^2; 
                    
                end
                
                
                
                if succ_flag_sub == 0
                    fail_no_sub = fail_no_sub + 1;
                end
                if succ_flag_sub == 2
                    diff_no_sub = diff_no_sub + 1;
                end
                if succ_flag_SOC == 0
                    fail_no_SOC = fail_no_SOC + 1;
                end
                if succ_flag_SOC == 2
                    diff_no_SOC = diff_no_SOC + 1;
                end
                if succ_flag_PAMAL == 0
                    fail_no_PAMAL = fail_no_PAMAL + 1;
                end
                if succ_flag_PAMAL == 2
                    diff_no_PAMAL = diff_no_PAMAL + 1;
                end

            end
            
            
            
            

            
            iter.manpg =  sum(maxit_att_manpg)/succ_no_manpg;
            iter.manpg_BB =  sum(maxit_att_manpg_BB)/succ_no_manpg_BB;
            iter.soc =  sum(maxit_att_soc)/succ_no_SOC;
            iter.pamal =  sum(maxit_att_pamal)/succ_no_PAMAL;
            iter.Rsub =  sum(maxit_att_Rsub)/succ_no_sub;
            iter.mialm =  sum(maxit_att_mialm)/succ_no_mialm;
            iter.admm =  sum(maxit_att_admm)/succ_no_admm;
            
            time.manpg =  sum(time_manpg)/succ_no_manpg;
            time.manpg_BB =  sum(time_manpg_BB)/succ_no_manpg_BB;
            time.soc =  sum(time_soc)/succ_no_SOC;
            time.pamal =  sum(time_pamal)/succ_no_PAMAL;
            time.Rsub =  sum(time_Rsub)/succ_no_sub;
            time.mialm =  sum(time_mialm)/succ_no_mialm;
            time.admm =  sum(time_admm)/succ_no_admm;
            
            Fval.manpg =  sum(F_manpg)/succ_no_manpg;
            Fval.manpg_BB =  sum(F_manpg_BB)/succ_no_manpg_BB;
            Fval.soc =  sum(F_soc)/succ_no_SOC;
            Fval.pamal =  sum(F_pamal)/succ_no_PAMAL;
            Fval.Rsub =  sum(F_Rsub)/succ_no_sub;
            %Fval.best = sum(F_best)/succ_no_manpg;
            Fval.mialm =  sum(F_mialm)/succ_no_mialm;
            Fval.admm =  sum(F_admm)/succ_no_admm;
            
            Sp.manpg =  sum(sparsity_manpg)/succ_no_manpg;
            Sp.manpg_BB =  sum(sparsity_manpg_BB)/succ_no_manpg_BB;
            Sp.soc =  sum(sparsity_soc)/succ_no_SOC;
            Sp.pamal =  sum(sparsity_pamal)/succ_no_PAMAL;
            Sp.Rsub =  sum(sparsity_Rsub)/succ_no_sub;
            Sp.mialm =  sum(sparsity_mialm)/succ_no_mialm;
            Sp.admm =  sum(sparsity_admm)/succ_no_admm;
            
%             residual.manpg =  sum(residual_manpg)/succ_no_manpg;
%             residual.manpg_BB =  sum(residual_manpg_BB)/succ_no_manpg_BB;
%             residual.soc =  sum(residual_soc)/succ_no_SOC;
%             residual.pamal =  sum(residual_PAMAL)/succ_no_PAMAL;
%             residual.Rsub =  sum(residual_Rsub)/succ_no_sub;
%             residual.mialm =  sum(residual_mialm)/succ_no_mialm;
%             residual.admm =  sum(residual_admm)/succ_no_admm;
            
             linesearch.mialm = sum(lins_mialm)/succ_no_mialm;
            linesearch.admm = sum(lins_admm)/succ_no_admm;
            linesearch.manpg = sum(lins_manpg)/succ_no_manpg;
            linesearch.manpg_BB = sum(lins_adap_manpg)/succ_no_manpg_BB;
            
            in_av.mialm = sum(in_av_mialm)/succ_no_mialm;
            in_av.admm = sum(in_av_admm)/succ_no_admm;
            in_av.manpg = sum(in_av_manpg)/succ_no_manpg;
            in_av.manpg_BB = sum(in_av_adap_manpg)/succ_no_manpg_BB;
            
            fprintf(fid,'==============================================================================================\n');
            % time
            A(1,1) = time.manpg;             A(1,2) = time.manpg_BB;     A(1,3) = time.Rsub;  A(1,4) = time.soc; 
            A(1,5) = time.pamal;             A(1,6) = time.mialm;          A(1,7) = time.admm;  
            
            % Fval
            A(2,1) = Fval.manpg;             A(2,2) = Fval.manpg_BB;     A(2,3) = Fval.Rsub;  A(2,4) = Fval.soc; 
            A(2,5) = Fval.pamal;             A(2,6) = Fval.mialm;         A(2,7) = Fval.admm;
            %sp
            A(3,1) = Sp.manpg;               A(3,2) = Sp.manpg_BB;       A(3,3) = Sp.Rsub;    A(3,4) = Sp.soc; 
            A(3,5) = Sp.pamal;               A(3,6) = Sp.mialm;           A(3,7) = Sp.admm;
            
%             %residual
%             A(4,1) = residual.manpg;               A(4,2) = residual.manpg_BB;       A(4,3) = residual.Rsub;    A(4,4) = residual.soc; 
%             A(4,5) = residual.pamal;               A(4,6) = residual.mialm;           A(4,7) = residual.admm;
%             
            A(5,1) = linesearch.manpg;             A(5,2) = linesearch.manpg_BB;     A(5,6) = linesearch.mialm;  A(5,7) = linesearch.admm;
            
            
            A(6,1) = in_av.manpg;                    A(6,2) = in_av.manpg_BB;        A(6,6) = in_av.mialm;      A(6,7) =  in_av.admm;
            
            A(7,1) = iter.manpg;                     A(7,2) = iter.manpg_BB;         A(7,3) = iter.Rsub;       A(7,4) = iter.soc;
            A(7,5) = iter.pamal;                      A(7,6)= iter.mialm;            A(7,7) = iter.admm;   
            
            A(8,1) = succ_no_manpg;                   A(8,2) = succ_no_manpg_BB;     A(8,3) = succ_no_sub;      A(8,4) = succ_no_SOC; 
            A(8,5) = succ_no_PAMAL;                   A(8,6) = succ_no_mialm;         A(8,7) = succ_no_admm; 
            
            info.C(:,:,id_n) = A;
            fprintf(fid,' Alg ****        Iter *****  Fval *** sparsity ** cpu *** Error ***\n');
            
            print_format =  'ManPG:      %1.3e  %1.5e    %1.2f      %3.2f \n';
            fprintf(fid,print_format, iter.manpg, Fval.manpg, Sp.manpg,time.manpg);
            print_format =  'ManPG_adap: %1.3e  %1.5e    %1.2f      %3.2f \n';
            fprintf(fid,print_format, iter.manpg_BB, Fval.manpg_BB, Sp.manpg_BB,time.manpg_BB);
            print_format =  'SOC:        %1.3e  %1.5e    %1.2f      %3.2f \n';
            fprintf(fid,print_format,iter.soc , Fval.soc, Sp.soc ,time.soc);
            print_format =  'PAMAL:      %1.3e  %1.5e    %1.2f      %3.2f \n';
            fprintf(fid,print_format,iter.pamal ,  Fval.pamal ,Sp.pamal,time.pamal);
            print_format =  'Rsub:       %1.3e  %1.5e    %1.2f      %3.2f  \n';
            fprintf(fid,print_format,iter.Rsub ,  Fval.Rsub ,Sp.Rsub,time.Rsub);
            print_format =  'Mmialm:       %1.3e  %1.5e    %1.2f      %3.5f  \n';
            fprintf(fid,print_format,iter.mialm ,  Fval.mialm ,Sp.mialm,time.mialm);
            print_format =  'ADMM:       %1.3e  %1.5e    %1.2f      %3.5f  \n';
            fprintf(fid,print_format,iter.admm ,  Fval.admm ,Sp.admm,time.admm);
        end
    end
end



end

