%function compare_spca
function  demo_compare_SPCA_realdata()
clear;
close all;
addpath ../misc
addpath ../SSN_subproblem
addpath ../
addpath ../util
addpath ../algorithm
addpath (genpath('../manopt'));

save_root = strcat('../results/spca/');
if ~exist(save_root,'dir')
    mkdir(save_root);
end
save_root_res = strcat(save_root,'res/');
if ~exist(save_root_res,'dir')
    mkdir(save_root_res);
end


data_root = strcat('../dataset/');
dirOutput = dir(fullfile(data_root,'*.mat'));
fileName = {dirOutput.name};



%format long
r_set = [10;20];   % rank
%r_set = [5;8;10;12;15];   % rank
%r_set = 5;

mu_set = [0.4;0.6];
%mu_set = [0.5;0.6;0.7;0.8];
%mu_set = 0.5;

%% problem setting

rng(1000);   test_num = 1;
table_str = '';  prob = 1:length(fileName);
for i = prob
    %% cycle
    base = fileName{i};
    B = load([data_root,base],'X');
    B = B.X; [m,n] = size(B);
    B = B - repmat(mean(B,1),m,1);
    B = normc(B);
    %B = B/sqrt(norm(B,'fro'));
    AtA = B'*B;
    for id_r =1:size(r_set,1) % r  number of column
        for id_mu = 1:size(mu_set,1)         % mu  sparse parameter
            r = r_set(id_r);
            lambda = mu_set(id_mu);
            
            basename = strrep(base,'.mat','');
            
            ret_manpg = zeros(test_num,7);    ret_mialm = zeros(test_num,9);
            ret_manpg_BB = zeros(test_num,7); ret_Rsub = zeros(test_num,5);
            ret_Soc = zeros(test_num,6);      ret_madmm = zeros(test_num,9);
            ret_pamal = zeros(test_num,6);
            
            for test_random = 1:test_num  %times average.
                type = 1;
                [phi_init,~] = svd(randn(n,r),0);  % random intialization
                
                option_Rsub.F_mialm = -1e10;
                option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 5e1;  option_Rsub.tol = 5e-3;
                option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=lambda;  option_Rsub.type = type;
                
                [phi_init]= Re_sub_spca(AtA,option_Rsub);
                
                
                A = struct();
                A.applyA = @(X) X;
                A.applyAT = @(y) y;
                
                f = struct();
                f.cost_grad = @pca_cost_grad;
                f.data = {AtA};
                
                h = struct();
                h.cost = @(X,lambda) lambda*sum(sum(abs(X)));
                h.prox = @(X,nu,lambda) max(abs(X) - nu*lambda,0).* sign(X);
                h.data = {lambda};
                
                
                manifold = stiefelfactory(n,r);
                
                
                %options_mialm.alpha = 1/(2*abs(eigs(full(AtA),1)));
                options_mialm.verbosity = 0;
                options_mialm.max_iter = 1000; options_mialm.tol = 1e-8*n*r;
                options_mialm.rho = 1.05;     options_mialm.tau = 0.8;
                options_mialm.nu0 = svds(AtA,1)^1 * 2;
                options_mialm.gtol0= 1e-0;   options_mialm.gtol_decrease = 0.8;
                options_mialm.X0 = phi_init;
                options_mialm.maxitersub = 10;
                options_mialm.verbosity = 1;
                
                [X_mialm,Z_mialm,out_mialm]=mialm(A, manifold, f, h, options_mialm);
                ret_mialm(test_random,:) = [out_mialm.obj, out_mialm.sparsity, out_mialm.time, out_mialm.iter, out_mialm.sub_iter, ...
                    out_mialm.deltak, out_mialm.etaD, out_mialm.etaC, out_mialm.nrmG];
                
                
                
                options_admm = options_mialm;
                options_admm.max_iter = 20000;  options_admm.opt = out_mialm.obj;
                options_admm.maxitersub = 100; %options_admm.nu0 = 500;
                
                [X_madmm,Z_madmm,out_madmm]=madmm(A, manifold, f, h, options_admm);
                ret_madmm(test_random,:) = [out_madmm.obj, out_madmm.sparsity, out_madmm.time, out_madmm.iter, out_madmm.sub_iter, ...
                    out_madmm.deltak, out_madmm.etaD, out_madmm.etaC, out_madmm.nrmG];
                
                
                
                %%%%%  manpg parameter
                option_manpg.opt = out_mialm.obj;
                option_manpg.adap = 0;    option_manpg.type = type;
                option_manpg.phi_init = phi_init; option_manpg.maxiter = 20000;  option_manpg.tol =1e-8*n*r;
                option_manpg.r = r;    option_manpg.n = n;  option_manpg.mu = lambda;
                option_manpg.inner_iter = 100;
                
                % profile on
                [X_manpg, F_manpg, sparsity_manpg,time_manpg,...
                    maxit_att_manpg, succ_flag_manpg, lins_manpg, in_av_manpg,obj_arr_manpg,time_arr_manpg]= manpg_orth_sparse(AtA,option_manpg);
                % profsave
                ret_manpg(test_random,:) = [F_manpg,sparsity_manpg, time_manpg, maxit_att_manpg, ...
                    succ_flag_manpg, lins_manpg, in_av_manpg];
                
                
                
                
                [X_manpg_BB, F_manpg_BB,sparsity_manpg_BB,time_manpg_BB,...
                    maxit_att_manpg_BB,succ_flag_manpg_BB,lins_adap_manpg,in_av_adap_manpg,obj_arr_manpg_BB,time_arr_manpg_BB]= manpg_orth_sparse_adap(AtA,option_manpg);
                
                ret_manpg_BB(test_random,:) = [F_manpg_BB,sparsity_manpg_BB,time_manpg_BB, maxit_att_manpg_BB, ...
                    succ_flag_manpg_BB,lins_adap_manpg,in_av_adap_manpg];
                
                
                
                
                
                
                
                %%%%%% Riemannian subgradient parameter
                option_Rsub.F_manpg = out_mialm.obj;
                option_Rsub.phi_init = phi_init; option_Rsub.maxiter = 2e4;      option_Rsub.tol = 5e-3;
                option_Rsub.r = r;    option_Rsub.n= n;  option_Rsub.mu=lambda;  option_Rsub.type = type;
                
                [X_Rsub, F_Rsub,sparsity_Rsub,time_Rsub,...
                    maxit_att_Rsub,succ_flag_sub,obj_arr_Rsub,time_arr_Rsub]= Re_sub_spca(AtA,option_Rsub);
                ret_Rsub(test_random,:) = [F_Rsub,sparsity_Rsub,time_Rsub, maxit_att_Rsub, succ_flag_sub];
                
                
                %%%%%% soc parameter
                option_soc.phi_init = phi_init; option_soc.maxiter = 20000;  option_soc.tol = 1e-9*n*r;
                option_soc.r = r;    option_soc.n = n;  option_soc.mu=lambda;
                %option_soc.L= L;
                option_soc.type = type;
                option_soc.F_mialm = out_mialm.obj;
                option_soc.X_mialm = X_mialm;
                
                [X_Soc, F_soc,sparsity_soc,time_soc,...
                    soc_error_XPQ,maxit_att_soc,succ_flag_SOC]= soc_spca(AtA,option_soc);
                ret_Soc(test_random,:) = [F_soc,sparsity_soc,time_soc, maxit_att_soc, ...
                    succ_flag_SOC,soc_error_XPQ];
                
                
                %%%%%% PAMAL parameter
                option_PAMAL.phi_init = phi_init; option_PAMAL.maxiter =2000;  option_PAMAL.tol = 1e-10*n*r;
                %option_PAMAL.L = L;   option_PAMAL.V = V;
                option_PAMAL.r = r;   option_PAMAL.n = n;  option_PAMAL.mu=lambda;   option_PAMAL.type = type;
                option_PAMAL.F_mialm = out_mialm.obj;
                option_PAMAL.X_mialm = X_mialm;
                
                [X_pamal, F_pamal,sparsity_pamal,time_pamal,...
                    pamal_error_XPQ, maxit_att_pamal,succ_flag_PAMAL]= PAMAL_spca(AtA,option_PAMAL);
                ret_pamal(test_random,:) = [F_pamal,sparsity_pamal,time_pamal, maxit_att_pamal, ...
                    succ_flag_PAMAL,pamal_error_XPQ];
                
                
                
                
                
            end
            
            save_path = strcat(save_root_res,basename,'_',num2str(r),'_',num2str(lambda),'.mat');
            save(save_path, 'ret_mialm', 'ret_madmm', 'ret_manpg', 'ret_manpg_BB', 'ret_Soc', 'ret_pamal', 'ret_Rsub', ...
                'X_mialm', 'X_madmm', 'Z_mialm', 'Z_madmm', 'X_manpg', 'X_manpg_BB', 'X_Rsub', 'X_Soc', 'X_pamal', 'n','r', 'lambda' );
            
            table_str = [table_str basename,'_',num2str(r),'_',num2str(lambda)];
            table_str = [table_str sprintf('& %.3e & %.3e & %.3e & %.3e & %.3e & %.3e & %.3e & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f', ...
                out_mialm.obj, out_madmm.obj,F_manpg,F_manpg_BB, F_soc, F_pamal, F_Rsub, out_mialm.time, out_madmm.time,time_manpg,time_manpg_BB, time_soc, time_pamal, time_Rsub)];
            table_str = [table_str '\\ \hline' newline];
        end
    end
end

disp(newline);
disp(table_str);
save_path = strcat(save_root,'spca_2.txt');
fid = fopen(save_path,'w+');
fprintf(fid,'%s',table_str);


    function [f,g] = pca_cost_grad(X,AtA)
        BX = AtA*X;
        f = -sum(sum(BX.*X));
        g = -2*BX;
    end
end

