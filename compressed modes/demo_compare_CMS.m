%function compare_spca
function info = demo_compare_CMS()
clc
clear;
close all;
addpath ../misc
addpath ../SSN_subproblem
%n_set=[1; 1500]; %dimension
n_set = 2.^(7:10);    %dimension
n_set = [128;256;400;512;600;700;800];
r_set = [2;4;6;8;10];   % rank
r_set = [4;5;6;7;8];   % rank

mu_set = [0.05;0.1;0.15;0.2;0.25;0.3];
index = 1;

%% Manifold parameter
problem.cost_f = @cost_f;
    function f = cost_f(X,BX)
         if ~exist('BX', 'var')  
             f = -sum(sum((H*X).*X));
         else
             f = -sum(sum(BX.*X));
         end
    end

problem.cost_g = @cost_g;
    function f = cost_g(X)
        f = lambda*sum(sum(abs(X)));
        %f = 0;
    end

problem.prox_g = @prox_g;
    function y = prox_g(X,mu)
        y = max(abs(X) - mu*lambda,0).* sign(X);
        %y = max(0,X);
    end


%% Riemannian gradient of the cost function
problem.egrad = @egrad;
    function g = egrad(X,BX)
         if ~exist('BX', 'var')  
            g = -2*H*X;
         else
             g = -2*BX;
         end
    end



%% cycle
info.n = 500; info.r = 5; info.C = zeros(7,7,length(n_set));
for id_n =  1:length(n_set)        % n  dimension
    n = n_set(id_n);
    fid =1;
    for id_r = 2% :size(r_set,1) % r  number of column
        for id_mu = 2%:length(mu_set)  %mu  sparsity parameter
            r = r_set(id_r);
            %mu = mu_set(id_mu);
            succ_no_manpg = 0;  succ_no_palm = 0;succ_no_sm = 0;succ_no_admm = 0;
            succ_no_manpg_BB = 0; succ_no_SOC = 0;  succ_no_PAMAL = 0; succ_no_sub = 0;
            diff_no_SOC = 0;  diff_no_PAMAL = 0;  diff_no_sub = 0;
            fail_no_SOC = 0;  fail_no_PAMAL = 0;  fail_no_sub = 0;
            A = zeros(7,7);
            for test_random =1:50  %times average.
                fprintf(fid,'==============================================================================================\n');
                
                
                L = 50; dx = L/n;  V = 0;
                %  phi_init = CMs_1D_initial(d,n,dx); % initial point ---Guass kernel
                H = -Sch_matrix(0,L,n); %  schrodinger operator
                %H = H/abs(eigs(H,1));
                lambda = mu_set(id_mu);
                fprintf(fid,'- n -- r -- mu --------\n');
                fprintf(fid,'%4d %3d %3.3f \n',n,r,lambda);
                fprintf(fid,'----------------------------------------------------------------------------------\n');
                problem.M = stiefelfactory(n, r);
                problem.B = 1;
                problem.AtA = H;
                
                
                
                
                
                rng('shuffle');
                [phi_init,~] = svd(randn(n,r),0);  % random intialization
                
                %% Remannian subgradient method
                option_Rsub.F_manpg = -1e10;
                option_Rsub.phi_init = phi_init; option_Rsub.maxiter = n*r;  option_Rsub.tol = 5e-3;
                option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=lambda;  option_Rsub.type = 1;
                
                [phi_init, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
                    maxit_att_Rsub(test_random),succ_flag_sub]= Re_sub_grad_spca(H,option_Rsub);
                
                
                Init = phi_init;
                options_palm.stepsize = 1/(2*abs(eigs(full(H),1)));
                options_palm.verbosity = 0;   options_palm.maxiter = 100;
                options_palm.tau = 0.99;      options_palm.rho = 1.05;
                options_palm.mu = 30/r^1/n;  
                %options_palm.mu = 1/(8/dx^2.*(sin(pi/4))^2 + V);
                %options_palm.mu = 1/(n*r*lambda + 1);
                %options_palm.mu = abs(1/eigs(H,1)^1) ;
                options_palm.k = r;
                options_palm.epso = 1e-5;     options_palm.iter = 5000;
                options_palm.tolgradnorm = 1; options_palm.decrease = 0.9;
                
                 options_palm.epso =1e-8*n*r;
%                 options_palm.tolgradnorm = 1e-1; 
%                 options_palm.rho = 1.05;      options_palm.decrease = 0.9;
%                 options_palm.maxiter = 10;   options_palm.epso = 1e-5;
                %
                [X_palm,F_palm(test_random),sparsity_palm(test_random),time_palm(test_random),...
                    maxit_att_palm(test_random),lins_palm(test_random),in_av_palm(test_random),succ_flag_palm] = Riemannian_mialm_CMS(problem, Init, options_palm);
                if succ_flag_palm == 1
                    succ_no_palm = succ_no_palm + 1;
                end
                
                
                
                options_admm = options_palm;          options_admm.opt = F_palm(test_random);
                %options_admm.mu = 0.01;
                %options_admm.mu = options_admm.mu/2; 
                options_admm.epso = 1e-5;
                options_admm.iter = 5000;              options_admm.tolgradnorm = 1e-5;
                options_admm.maxiter = 100;            options_admm.epso = 1e-8*n*r;
                [X_admm,F_admm(test_random),sparsity_admm(test_random),time_admm(test_random),...
                    maxit_att_admm(test_random),lins_admm(test_random),in_av_admm(test_random),succ_flag_admm] = Riemannian_admm_CMS(problem, Init, options_admm);
                if succ_flag_admm == 1
                    succ_no_admm = succ_no_admm + 1;
                end
                
                
                
                %%  manpg
                %option_manpg.adap = 0;
                option_manpg.phi_init = phi_init; option_manpg.maxiter = 30000;  option_manpg.tol =1e-8*n*r;
                option_manpg.r = r;    option_manpg.n = n;  option_manpg.mu = lambda;
                option_manpg.L = L; option_manpg.F_palm = F_palm(test_random);
                %option_manpg.inner_tol =1e-11;
                option_manpg.inner_iter = 100;
                %% soc parameter
                option_soc.phi_init = phi_init; option_soc.maxiter = 30000;  option_soc.tol =1e-5;
                option_soc.r = r;    option_soc.n = n;  option_soc.mu=lambda;
                option_soc.L= L;
                %% PAMAL parameter
                option_PAMAL.phi_init = phi_init; option_PAMAL.maxiter = 30000;  option_PAMAL.tol =1e-5;
                option_PAMAL.L = L;   option_PAMAL.V = V;
                option_PAMAL.r = r;   option_PAMAL.n = n;  option_PAMAL.mu=lambda;
                
                [X_manpg, F_manpg(test_random),sparsity_manpg(test_random),time_manpg(test_random),...
                    maxit_att_manpg(test_random),succ_flag_manpg,lins_manpg(test_random),in_av_manpg(test_random)]= manpg_CMS(H,option_manpg,dx,V);
                if succ_flag_manpg == 1
                    succ_no_manpg = succ_no_manpg + 1;
                    
                end
                
                %option_manpg.F_manpg = F_palm(test_random);
                [X_manpg_BB, F_manpg_BB(test_random),sparsity_manpg_BB(test_random),time_manpg_BB(test_random),...
                    maxit_att_manpg_BB(test_random),succ_flag_manpg_BB,lins_adap_manpg(test_random),in_av_adap_manpg(test_random)]= manpg_CMS_adap(H,option_manpg,dx,V);
                if succ_flag_manpg_BB == 1
                    succ_no_manpg_BB = succ_no_manpg_BB + 1;
               elseif(succ_flag_palm == 1)
                    time_palm(test_random) = 0;
                    F_palm(test_random) = 0;
                    sparsity_palm(test_random) = 0;
                    maxit_att_palm(test_random) = 0;
                    lins_palm(test_random) = 0;
                    in_av_palm(test_random) = 0;
                    succ_no_palm = succ_no_palm  - 1;
                    
                end
                
                %% Riemannian subgradient parameter
                option_Rsub.F_manpg = F_manpg(test_random);
                option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 1e1;  option_Rsub.tol = 1e-3;
                option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=lambda;  option_Rsub.type = 1;
                
                [X_Rsub, F_Rsub(test_random),sparsity_Rsub(test_random),time_Rsub(test_random),...
                    maxit_att_Rsub(test_random),succ_flag_sub]= Re_sub_grad_spca(H,option_Rsub);
                %phi_init = X_Rsub;
                if succ_flag_sub == 1
                    succ_no_sub = succ_no_sub + 1;
                end
                
                
                option_soc.F_palm = F_palm(test_random);
                option_soc.X_palm = X_palm;
                option_PAMAL.F_palm = F_palm(test_random);
                option_PAMAL.X_palm = X_palm;
                %option_soc.beta = 1/svds(H,1)/1 ;
                [X_Soc, F_soc(test_random),sparsity_soc(test_random),time_soc(test_random),...
                    soc_error_XPQ(test_random),maxit_att_soc(test_random),succ_flag_SOC]= soc_CM(H,option_soc);
                if succ_flag_SOC == 1
                    succ_no_SOC = succ_no_SOC + 1;
                end
                
               % option_PAMAL.beta = 1/svds(H,1) ;
                [X_pamal, F_pamal(test_random),sparsity_pamal(test_random),time_pamal(test_random),...
                    pam_error_XPQ(test_random), maxit_att_pamal(test_random),succ_flag_PAMAL]= PAMAL_CMs(H,option_PAMAL,V);
                if succ_flag_PAMAL ==1
                    succ_no_PAMAL = succ_no_PAMAL + 1;
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
            
            
            
            
            
           
            index = index +1;
            
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
            %Fval.best = sum(F_best)/succ_no_manpg;
            Fval.admm =  sum(F_admm)/succ_no_admm;
            Fval.palm =  sum(F_palm)/succ_no_palm;
            
            Sp.manpg =  sum(sparsity_manpg)/succ_no_manpg;
            Sp.manpg_BB =  sum(sparsity_manpg_BB)/succ_no_manpg_BB;
            Sp.soc =  sum(sparsity_soc)/succ_no_SOC;
            Sp.pamal =  sum(sparsity_pamal)/succ_no_PAMAL;
            Sp.Rsub =  sum(sparsity_Rsub)/succ_no_sub;
            Sp.palm =  sum(sparsity_palm)/succ_no_palm;
            Sp.admm =  sum(sparsity_admm)/succ_no_admm;
            
            linesearch.palm = sum(lins_palm)/succ_no_palm;
            linesearch.admm = sum(lins_admm)/succ_no_admm;
            linesearch.manpg = sum(lins_manpg)/succ_no_manpg;
            linesearch.manpg_BB = sum(lins_adap_manpg)/succ_no_manpg_BB;
            
            in_av.palm = sum(in_av_palm)/succ_no_palm;
            in_av.admm = sum(in_av_admm)/succ_no_admm;
            in_av.manpg = sum(in_av_manpg)/succ_no_manpg;
            in_av.manpg_BB = sum(in_av_adap_manpg)/succ_no_manpg_BB;
            
            
            % time
            A(1,1) = time.manpg;             A(1,2) = time.manpg_BB;      A(1,3) = time.Rsub;  A(1,4) = time.soc;
            A(1,5) = time.pamal;             A(1,6) = time.palm;          A(1,7) = time.admm;
            
            % Fval
            A(2,1) = Fval.manpg;             A(2,2) = Fval.manpg_BB;     A(2,3) = Fval.Rsub;  A(2,4) = Fval.soc;
            A(2,5) = Fval.pamal;             A(2,6) = Fval.palm;         A(2,7) = Fval.admm;
            %sp
            A(3,1) = Sp.manpg;               A(3,2) = Sp.manpg_BB;       A(3,3) = Sp.Rsub;    A(3,4) = Sp.soc;
            A(3,5) = Sp.pamal;               A(3,6) = Sp.palm;           A(3,7) = Sp.admm;
            
             A(5,1) = linesearch.manpg;               A(5,2) = linesearch.manpg_BB;        A(5,6) = linesearch.palm;  A(5,7) = linesearch.admm;
            
            
            A(6,1) = in_av.manpg;                    A(6,2) = in_av.manpg_BB;             A(6,6) = in_av.palm;      A(6,7) =  in_av.admm;
            
            
            A(7,1) = iter.manpg;                     A(7,2) = iter.manpg_BB;         A(7,3) = iter.Rsub;       A(7,4) = iter.soc;
            A(7,5) = iter.pamal;                      A(7,6)= iter.palm;            A(7,7) = iter.admm;   
            
            A(4,1) = succ_no_manpg;                   A(4,2) = succ_no_manpg_BB;     A(4,3) = succ_no_sub;      A(4,4) = succ_no_SOC; 
            A(4,5) = succ_no_PAMAL;                   A(4,6) = succ_no_palm;         A(4,7) = succ_no_admm; 
            
           
            
            info.C(:,:,id_n) = A;
            
            fprintf(fid,'==============================================================================================\n');
            
            fprintf(fid,' Alg *****      Iter *****  Fval *** sparsity ** cpu *** Error ***\n');
            
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
            print_format =  'PALM:       %1.3e  %1.5e    %1.2f      %3.2f  \n';
            fprintf(fid,print_format,iter.palm ,  Fval.palm ,Sp.palm,time.palm);
            print_format =  'ADMM:       %1.3e  %1.5e    %1.2f      %3.2f  \n';
            fprintf(fid,print_format,iter.admm ,  Fval.admm ,Sp.admm,time.admm);
        end
    end
end




end
