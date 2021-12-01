function [X_Rsub,F_Sub,sparsity,time_Rsub,i,succ_flag,f_re_sub,time_arr] = Re_sub_grad_spca(B,option)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% remannian subgradient;
%min -Tr(X'*B*X)+ mu*norm(X,1) s.t. X'*X=Ir. X \in R^{p*r}
tt = tic;
r=option.r;
n=option.n;
mu = option.mu;
maxiter =option.maxiter + 1;
tol = option.tol;
stepsize = 1/svds(B,1)^2*2;
X = option.phi_init;

if option.type == 1
    AX = B*X;
else
    AX = B'*(B*X);
end

h=@(X) mu*sum(sum(abs(X)));

f_re_sub = zeros(maxiter,1); 
time_arr = zeros(maxiter,1); 
succ_flag = 0;
f_re_sub(1) = -sum(sum(X.*(AX))) + h(X);
for i = 2:maxiter
    X0 = X;
    gx = 2*AX - mu*sign(X0) ; %negative Euclidean gradient
    xgx = X0'*gx;
    pgx = gx - 0.5*X0*(xgx+xgx');   %negative  Riemannian gradient using Euclidean metric
    %pgx = gx;
    eta = stepsize*1^i; 
    %eta = 1/i^(3/4);  
    %eta = 1/i;  
    X = X0 + eta * pgx;    % Riemannian step
    %[q,~] = qr(q);    % retraction
    [U, SIGMA, S] = svd(X'*X);   SIGMA =diag(SIGMA);    X = X*(U*diag(sqrt(1./SIGMA))*S');
    Fnew = -sum(sum(X.*(B*X))) + h(X); iter  =1;
    while Fnew > f_re_sub(i-1) && iter < 3
        X = X0 + eta*0.5^iter * pgx;    % Riemannian step
    %[q,~] = qr(q);    % retraction
    [U, SIGMA, S] = svd(X'*X);   SIGMA =diag(SIGMA);    X = X*(U*diag(sqrt(1./SIGMA))*S');
    Fnew = -sum(sum(X.*(B*X))) + h(X);
    iter = iter +1;
    end
    if option.type == 1
        AX = B*X;
    else
        AX = B'*(B*X);
    end
    
    f_re_sub(i) = Fnew;
    time_arr(i) = toc(tt);
    if  f_re_sub(i) < option.F_mialm + 1e-4
        succ_flag = 1;
        break;
    end
   
end
X((abs(X)<=1e-5))=0;
X_Rsub = X;
time_Rsub = toc(tt);
sparsity= sum(sum(X_Rsub==0))/(n*r);
F_Sub = f_re_sub(i-1);
 
%plot(f_re_sub);
%hold on;
%plot(norm_grad);
   fprintf('Rsub:Iter ***  Fval *** CPU  **** sparsity \n');
    
    print_format = ' %i     %1.5e    %1.2f     %1.2f    \n';
    fprintf(1,print_format, i-1, F_Sub, time_Rsub, sparsity);
end

