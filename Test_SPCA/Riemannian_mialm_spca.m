function [X,F,sp,times,t,num_linesearch,num_inniter,flag] = Riemannian_mialm_spca(Problem, X0, option)



cost_f = Problem.cost_f; cost_g = Problem.cost_g; prox_g = Problem.prox_g;
grad_f = Problem.egrad; %ehess_f = Problem.ehess;
if(isfield(Problem, 'partialegrad'))
    partialgrad_f = Problem.partialegrad;
end
B = Problem.B;


tau = option.tau; rho = option.rho; mu = option.mu;


epso = option.epso;k = option.k;
%% Read dataset



problem.M = Problem.M;  problem.AtA = Problem.AtA;
if(isfield(Problem, 'ncostterms'))
    problem.ncostterms = Problem.ncostterms;
end




%% Cost function
problem.cost = @cost;

    function f = cost(X,BX)
        
        temp = B*X-mu*Z;
        Prox = prox_g(temp,mu);
         if ~exist('BX', 'var')  
             f = cost_f(X) + cost_g(Prox) + 1/(2*mu) * norm(temp -Prox,'fro' )^2;
         else
             f = cost_f(X,BX) + cost_g(Prox) + 1/(2*mu) * norm(temp -Prox,'fro' )^2;
         end
    end



%% Riemannian gradient of the cost function
problem.egrad = @egrad;
    function g = egrad(X,BX)
        Prox = prox_g(X-mu*Z,mu);
         if ~exist('BX', 'var')  
             g = grad_f(X) + 1/mu * (X-mu*Z -Prox);
         else
             g = grad_f(X,BX) + 1/mu * (X-mu*Z -Prox);
    
         end
    end

problem.ehess = @ehess;
    function g = ehess(X,U)
        temp = abs(prox_g(X-mu*Z,mu));
        for i=1:n
            for j=1:k
                if(temp(i,j)~=0)
                    temp(i,j) = 1;
                end
            end
        end
        g = ehess_f(X,U) + 1/mu*((ones(size(B,1),k) - temp).*U);
    end



%% Riemannian stochastic gradient of the cost function
problem.partialegrad = @partialegrad;
    function g = partialegrad(X,idx_batchsize)
        Prox = prox_g(B*X-mu*Z,mu);
        g = partialgrad_f(X,idx_batchsize) + 1/mu * B'*(B*X-mu*Z -Prox);
        
    end


Z = zeros(size(B*X0));

info_err = zeros(option.maxiter,1); t = 0;
info_inniter = zeros(option.maxiter,1);
info_line = zeros(option.maxiter,1);
%F = zeros(option.maxiter,1);
temp_err0 = 0;
option.flag = 0;

 tic;
while(t<option.iter)
    %option.tolgradnorm = max(1e-6,option.tolgradnorm*option.decrease);
    option.tolgradnorm = max(1e-4,0.9^t);
    [X,info] = steepest_mialm_spca(problem,X0,option);
   % [X,info] = conjugategradient_mialm(problem,X0,option);
    
    option.flag = 1;
    
    Y = prox_g(X-mu*Z,mu);
    Z = Z - 1/mu * (X-Y);
    
    
    % update mu
    temp_err = norm(X-Y,'fro');
    if(t>0 && temp_err>=tau*temp_err0)
        mu = max(mu/rho,option.mu/10);
    end
    temp_err0 = temp_err;
    
    
    
    
    
    
    
    % KKT condition
    kkt_X = grad_f(X,info(end).BX) - Z;
    %kkt_X = grad_f(X) - Z;
    xgx = X'*kkt_X;
    kkt_1 = norm(kkt_X - 0.5*X*(xgx+xgx'), 'fro')^2;
    kkt_1 = kkt_1/(1+kkt_1);
   % kkt_1 = kkt_1/(1+norm(Z,'fro')^2);
    
    
    %kkt_2 = norm(X + prox_g(Z - X,1),'fro')^2;
    kkt_2 = norm(X - prox_g(X - 1*Z,1), 'fro')^2;
    kkt_2 = kkt_2/(1+kkt_2);
    %kkt_2 = kkt_2/(1+norm(Z,'fro')^2+norm(X,'fro')^2);
    
    info_err(t+1) = max(kkt_1,kkt_2);
     if(max(kkt_1,kkt_2)<epso)% || F(t)<option.opt)
        break;
    end
    t = t+1;
    X0 = X;
   
    
    option.BX = info(end).BX;
    %option.desc_dir = info(end).desc_dir;
    
    %information

    
    info_inniter(t) = info(end).iter;
    info_line(t) = info(end).num_linesearch;
    
    option.stepsize = info.stepsize;
end


if(t==option.iter)
    F = 0;
    sp = 0;
    times = 0;
    flag = 0;
    t = 0;
    num_linesearch = 0;
    num_inniter = 0;
    
else
    
    
    
    F = cost_f(X) + cost_g(X);
    
    [n,k] = size(Y); tol = max(max(X))/1e4;
    sp = sum(sum(abs(X)<1e-4))/(n*k);
    num_inniter = sum(info_inniter)/t;
    num_linesearch = sum(info_line);    
    flag = 1;
    times = toc;
end
%option.err = info_err; option.time = info_time; option.inniter = info_inniter;



fprintf('mialm:Iter ***  Fval *** CPU  **** sparsity ***        iaverge_No.   ** err ***   inner_opt  \n');

print_format = ' %i     %1.5e    %1.6f       %1.2f                %2.2f         %1.3e    %1.3e   \n';

fprintf(1,print_format, t,F,times, sp, sum(info_inniter)/t, info_err(t),info(end).gradnorm);

end


