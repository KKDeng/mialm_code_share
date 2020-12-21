function [X,F,sp,times,t,num_linesearch,num_inniter,flag] = Riemannian_admm_spca(Problem, X0, option)
%%   riemannian admm
%           f(X) + g(BX)
%           f(X) + g(Y) -<BX-Y,Z> + 1/mu||BX-Y||^2
% 


cost_f = Problem.cost_f;  prox_g = Problem.prox_g;
grad_f = Problem.egrad; cost_g = Problem.cost_g;
B = Problem.B; 


 mu = option.mu;
epso = option.epso;k = option.k;
%% Read dataset



problem.M = Problem.M;
%problem.ncostterms = Problem.ncostterms;



%% Cost function
problem.cost = @cost;
    function f = cost(X)
        f = cost_f(X) + 1/(2*mu) * norm(B*X- Y - mu*Z,'fro' )^2;        
    end



%% Riemannian gradient of the cost function
problem.egrad = @egrad;
    function g = egrad(X)
        g = grad_f(X) + 1/mu * B'*(B*X-mu*Z -Y);
    end

n = size(X0,1);




 Y = B*X0;Z = zeros(size(Y));

info_err = zeros(option.iter,1); t = 0;
info_inniter = zeros(option.iter,1);
info_line = zeros(option.iter,1);
time0 =  tic;
while(t<option.iter)
    
    %[X,~,info,~] = barzilaiborwein(problem,X0,option);
    [X,info] = steepest(problem,X0,option);
    
    Y = prox_g(B*X-mu*Z,mu);
    Z = Z - 1/mu * (B*X-Y);
   
    X0 = X;  
    t = t+1;
    
    F = cost_f(X) + cost_g(X);
    
    kkt_X = grad_f(X) - Z;
    xgx = X'*kkt_X;
    kkt_1 = norm(kkt_X - 0.5*X*(xgx+xgx'))^2;
    
    kkt_2 = norm(X + prox_g(Z - X,1))^2;
    if(max(kkt_1,kkt_2)<epso && F<=option.opt + 1e-7)
        break;
    end
    info_err(t) = max(kkt_1,kkt_2);
    info_inniter(t) = info(end).iter;
    info_line(t) = info(end).num_linesearch;
   
end


% if t == option.iter && max(kkt_1,kkt_2) > 1e-1
%     flag = 0;
%     sp  = 0;
%     F = 0;
%     times = 0;
% end

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
    
    sp = sum(sum(abs(X)<=1e-4))/(n*k);
    
    times = toc(time0);
    num_inniter = sum(info_inniter)/t;
    num_linesearch = sum(info_line);
    flag = 1;
    
end




fprintf('MADMM:Iter ***  Fval *** CPU  **** sparsity ***        iaverge_No.   ** err ***   inner_opt  \n');
                
print_format = ' %i     %1.5e    %1.2f       %1.2f                %2.2f         %1.3e    %1.3e   \n';
       
fprintf(1,print_format, t,F,times, sp, num_inniter, max(kkt_1,kkt_2),info(end).gradnorm);

end
