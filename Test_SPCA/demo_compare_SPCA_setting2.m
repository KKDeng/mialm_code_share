%function compare_spca
function info = demo_compare_SPCA_setting2()
clear;
close all;
addpath ../misc
addpath ../SSN_subproblem
n_set=[ 300; 400; 500; ]; %dimension  
%n_set = 500;
%format long
r_set = [5;8;10;15];   % rank
%r_set = 5

mu_set = [0.5;0.5;0.8];
%mu_set = 0.5;

%% problem setting
problem.cost_f = @cost_f;
    function f = cost_f(X,BX)
        if ~exist('BX', 'var')
            BX = AtA*X;
        end
        f = -sum(sum(BX.*X));
    end

problem.cost_g = @cost_g;
    function f = cost_g(X)
        f = lambda*sum(sum(abs(X)));
    end


problem.prox_g = @prox_g;
    function y = prox_g(X,mu)
        y = max(abs(X) - mu*lambda,0).* sign(X);
    end

problem.egrad = @egrad;
    function g = egrad(X,BX)
        if ~exist('BX', 'var')   
            BX = AtA*X;
        end
        g = -2*BX;
    end

problem.ehess = @ehess;
    function g = ehess(X,U)
        g = -(AtA*U);
    end

info.n = 500; info.r = 5; info.C = zeros(8,7,length(r_set),length(n_set));
%% cycle
for id_n = 1:size(n_set,1)        % n  dimension
    n = n_set(id_n);
    fid =1;    
    for id_r = 1:size(r_set,1) % r  number of column
        for id_mu = 2%:size(mu_set,1)         % mu  sparse parameter
            r = r_set(id_r);
            lambda = mu_set(id_mu);
            succ_no_manpg = 0;    succ_no_palm = 0;     succ_no_admm = 0; 
            succ_no_manpg_BB = 0; succ_no_SOC = 0;  succ_no_PAMAL = 0; succ_no_sub = 0;
            diff_no_SOC = 0;  diff_no_PAMAL = 0;  diff_no_sub = 0;
            fail_no_SOC = 0;  fail_no_PAMAL = 0;  fail_no_sub = 0;
            residual_manpg = zeros(50,1);   residual_palm = zeros(50,1);  residual_manpg_BB = zeros(50,1);  residual_admm = zeros(50,1);  
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
                s = (1:n/5);
                u= zeros(n,r); v = rand(n,r); v = v/sqrtm(v'*v);
                u(s,:) = randi( [-2, 2], size(u(s,:)));
                u = u/sqrtm(u'*u);
                
                Lam = diag(0.9*ones(1,r));
                if(r == 2)
                    Lam = diag( [0.9;0.8] );
                end
                
                if(r == 4)
                    Lam = diag( [0.95;0.9;0.85;0.8] );
                end  
                
                if(r == 3)
                    Lam = diag( [0.9;0.85;0.8] );
                end  
                
                
                    
                sigma = u*Lam*u'+ 0.01*(v*v');
                %sigma = v*Lam*v';
                B = mvnrnd(zeros(n,1),sigma,m);
                %B = normc(B);
                
                
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
                
                
                Init = phi_init;   %options_palm.opt = F_manpg(test_random);
                options_palm.stepsize = 1/(2*abs(eigs(full(AtA),1)));
                options_palm.iter = 5000;    options_palm.verbosity = 0;
                options_palm.maxiter = 100; options_palm.epso = 1e-8*n*r;
                options_palm.tau = 0.99;    options_palm.rho = 1.05;
                options_palm.k = r;    
                options_palm.mu = 0.5/svds(AtA,1)^1 ;
                options_palm.tolgradnorm = 1; options_palm.decrease = 0.9;
                options_palm.AtA = AtA;       options_palm.beta_type = 'P-R';
                
                 options_palm.maxiter = 100;    
               
                
                 

                [X_palm,F_palm(test_random),sparsity_palm(test_random),time_palm(test_random),...
                    maxit_att_palm(test_random),lins_palm(test_random),in_av_palm(test_random),succ_flag_palm] = Riemannian_mialm_spca(problem, Init,options_palm);

                if succ_flag_palm == 1
                    succ_no_palm = succ_no_palm + 1;
                    residual_palm(test_random) = norm(u*u' - X_palm*X_palm','fro')^2;  
                end
                
                
                
                
               
                
                
                
                %%%%%  manpg parameter
                option_manpg.opt = F_palm(test_random);
                option_manpg.adap = 0;    option_manpg.type =type;
                option_manpg.phi_init = phi_init; option_manpg.maxiter = 10000;  option_manpg.tol =1e-8*n*r;
                option_manpg.r = r;    option_manpg.n = n;  option_manpg.mu = lambda;
                %option_manpg.L = L;
                %option_manpg.inner_tol =1e-11;
                option_manpg.inner_iter = 100;
                %%%%%% soc parameter
                option_soc.phi_init = phi_init; option_soc.maxiter = 20000;  option_soc.tol =1e-5;
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
                    residual_manpg(test_random) = norm(u*u' - X_manpg*X_manpg','fro')^2; 
                end
                
                

                
                option_manpg.F_manpg = F_palm(test_random);
                [X_manpg_BB, F_manpg_BB(test_random),sparsity_manpg_BB(test_random),time_manpg_BB(test_random),...
                    maxit_att_manpg_BB(test_random),succ_flag_manpg_BB,lins_adap_manpg(test_random),in_av_adap_manpg(test_random)]= manpg_orth_sparse_adap(B,option_manpg);
                if succ_flag_manpg_BB == 1
                    succ_no_manpg_BB = succ_no_manpg_BB + 1;
                    residual_manpg_BB(test_random) = norm(u*u' - X_manpg_BB*X_manpg_BB','fro')^2; 
                    
                elseif(succ_flag_palm == 1)
                    time_palm(test_random) = 0;
                    F_palm(test_random) = 0;
                    sparsity_palm(test_random) = 0;
                    maxit_att_palm(test_random) = 0;
                    lins_palm(test_random) = 0;
                    in_av_palm(test_random) = 0;
                    succ_no_palm = succ_no_palm  - 1;
                end
                
                
                
                
                
                
                
                %%%%%% Riemannian subgradient parameter
                option_Rsub.F_manpg = F_manpg(test_random);
                option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 1e4;      option_Rsub.tol = 5e-3;
                option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=lambda;  option_Rsub.type = type;
                
                [X_Rsub, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
                    maxit_att_Rsub(test_random),succ_flag_sub]= Re_sub_grad_spca(B,option_Rsub);
                %phi_init = X_Rsub;
                if succ_flag_sub == 1
                    succ_no_sub = succ_no_sub + 1;
                    residual_Rsub(test_random) = norm(u*u' - X_Rsub*X_Rsub','fro')^2; 
                end
                option_soc.F_palm = F_palm(test_random);
                option_soc.X_palm = X_palm;
                option_PAMAL.F_palm = F_palm(test_random);
                option_PAMAL.X_palm = X_palm;
                [X_Soc, F_soc(test_random),sparsity_soc(test_random),time_soc(test_random),...
                    soc_error_XPQ(test_random),maxit_att_soc(test_random),succ_flag_SOC]= soc_spca(B,option_soc);
               % succ_flag_SOC = 1;
                if succ_flag_SOC == 1
                    succ_no_SOC = succ_no_SOC + 1;
                    residual_soc(test_random) = norm(u*u' - X_Soc*X_Soc','fro')^2; 
                end
                option_PAMAL.F_manpg = F_palm(test_random);
                [X_pamal, F_pamal(test_random),sparsity_pamal(test_random),time_pamal(test_random),...
                    pam_error_XPQ(test_random), maxit_att_pamal(test_random),succ_flag_PAMAL]= PAMAL_spca1(B,option_PAMAL);
               % succ_flag_PAMAL = 1;
                if succ_flag_PAMAL ==1
                    succ_no_PAMAL = succ_no_PAMAL + 1;
                    residual_PAMAL(test_random) = norm(u*u' - X_pamal*X_pamal','fro')^2; 
                end
                
                
                
                %opt = min(F_palm,F_manpg);
                
                
                
                
                options_admm = options_palm;
                options_admm.mu = 0.5/svds(AtA,1)^1;
                options_admm.iter = 5000;  options_admm.opt = F_palm(test_random);
                options_admm.maxiter = 100;
                options_admm.tolgradnorm = 1e-4;
                
                [X_admm,F_admm(test_random),sparsity_admm(test_random),time_admm(test_random),...
                    maxit_att_admm(test_random),lins_admm(test_random),in_av_admm(test_random),succ_flag_admm] = Riemannian_admm_spca(problem, Init, options_admm);
                if succ_flag_admm == 1
                    succ_no_admm = succ_no_admm + 1;
                    residual_admm(test_random) = norm(u*u' - X_admm*X_admm','fro')^2; 
                    
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
            iter.palm =  sum(maxit_att_palm)/succ_no_palm;
            iter.admm =  sum(maxit_att_admm)/succ_no_admm;
            
            time.manpg =  sum(time_manpg)/succ_no_manpg;
            time.manpg_BB =  sum(time_manpg_BB)/succ_no_manpg_BB;
            time.soc =  sum(time_soc)/succ_no_SOC;
            time.pamal =  sum(time_pamal)/succ_no_PAMAL;
            time.Rsub =  sum(time_Rsub)/succ_no_sub;
            time.palm =  sum(time_palm)/succ_no_palm;
            time.admm =  sum(time_admm)/succ_no_admm;
            
            Fval.manpg =  sum(F_manpg)/succ_no_manpg;
            Fval.manpg_BB =  sum(F_manpg_BB)/succ_no_manpg_BB;
            Fval.soc =  sum(F_soc)/succ_no_SOC;
            Fval.pamal =  sum(F_pamal)/succ_no_PAMAL;
            Fval.Rsub =  sum(F_Rsub)/succ_no_sub;
            Fval.palm =  sum(F_palm)/succ_no_palm;
            Fval.admm =  sum(F_admm)/succ_no_admm;
            
            Sp.manpg =  sum(sparsity_manpg)/succ_no_manpg;
            Sp.manpg_BB =  sum(sparsity_manpg_BB)/succ_no_manpg_BB;
            Sp.soc =  sum(sparsity_soc)/succ_no_SOC;
            Sp.pamal =  sum(sparsity_pamal)/succ_no_PAMAL;
            Sp.Rsub =  sum(sparsity_Rsub)/succ_no_sub;
            Sp.palm =  sum(sparsity_palm)/succ_no_palm;
            Sp.admm =  sum(sparsity_admm)/succ_no_admm;
            
            residual.manpg =  sum(residual_manpg)/succ_no_manpg;
            residual.manpg_BB =  sum(residual_manpg_BB)/succ_no_manpg_BB;
            residual.soc =  sum(residual_soc)/succ_no_SOC;
            residual.pamal =  sum(residual_PAMAL)/succ_no_PAMAL;
            residual.Rsub =  sum(residual_Rsub)/succ_no_sub;
            residual.palm =  sum(residual_palm)/succ_no_palm;
            residual.admm =  sum(residual_admm)/succ_no_admm;
            
            linesearch.palm = sum(lins_palm)/succ_no_palm;
            linesearch.admm = sum(lins_admm)/succ_no_admm;
            linesearch.manpg = sum(lins_manpg)/succ_no_manpg;
            linesearch.manpg_BB = sum(lins_adap_manpg)/succ_no_manpg_BB;
            
            in_av.palm = sum(in_av_palm)/succ_no_palm;
            in_av.admm = sum(in_av_admm)/succ_no_admm;
            in_av.manpg = sum(in_av_manpg)/succ_no_manpg;
            in_av.manpg_BB = sum(in_av_adap_manpg)/succ_no_manpg_BB;
           
            
            
            
            fprintf(fid,'==============================================================================================\n');
            % time
            A(1,1) = time.manpg;             A(1,2) = time.manpg_BB;     A(1,3) = time.Rsub;  A(1,4) = time.soc; 
            A(1,5) = time.pamal;             A(1,6) = time.palm;             A(1,7) = time.admm;  
            
            % Fval
            A(2,1) = Fval.manpg;             A(2,2) = Fval.manpg_BB;     A(2,3) = Fval.Rsub;  A(2,4) = Fval.soc; 
            A(2,5) = Fval.pamal;             A(2,6) = Fval.palm;            A(2,7) = Fval.admm;
            %sp
            A(3,1) = Sp.manpg;               A(3,2) = Sp.manpg_BB;       A(3,3) = Sp.Rsub;    A(3,4) = Sp.soc; 
            A(3,5) = Sp.pamal;               A(3,6) = Sp.palm;                A(3,7) = Sp.admm;
            
            %residual
            A(4,1) = residual.manpg;               A(4,2) = residual.manpg_BB;       A(4,3) = residual.Rsub;    A(4,4) = residual.soc; 
            A(4,5) = residual.pamal;               A(4,6) = residual.palm;                A(4,7) = residual.admm;
            
            A(5,1) = linesearch.manpg;               A(5,2) = linesearch.manpg_BB;        A(5,6) = linesearch.palm;  A(5,7) = linesearch.admm;
            
            
            A(6,1) = in_av.manpg;                    A(6,2) = in_av.manpg_BB;             A(6,6) = in_av.palm;      A(6,7) =  in_av.admm;
            
            A(7,1) = iter.manpg;                     A(7,2) = iter.manpg_BB;         A(7,3) = iter.Rsub;       A(7,4) = iter.soc;
            A(7,5) = iter.pamal;                      A(7,6)= iter.palm;            A(7,7) = iter.admm;   
            
            A(8,1) = succ_no_manpg;                   A(8,2) = succ_no_manpg_BB;     A(8,3) = succ_no_sub;      A(8,4) = succ_no_SOC; 
            A(8,5) = succ_no_PAMAL;                   A(8,6) = succ_no_palm;         A(8,7) = succ_no_admm; 
            
            
            info.C(:,:,id_r,id_n) = A;
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
            print_format =  'MPALM:       %1.3e  %1.5e    %1.2f      %3.5f  \n';
            fprintf(fid,print_format,iter.palm ,  Fval.palm ,Sp.palm,time.palm);
            print_format =  'ADMM:       %1.3e  %1.5e    %1.2f      %3.5f  \n';
            fprintf(fid,print_format,iter.admm ,  Fval.admm ,Sp.admm,time.admm);
        end
    end
end



end

