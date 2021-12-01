function [B, ret] = R_sub_ODL(Xtilde, Bo, opts)

if ~isfield(opts, 'max_iter'); opts.max_iter = 1e3; end
if ~isfield(opts, 'beta'); opts.beta = 0.9; end
if ~isfield(opts, 'mu0'); opts.mu_0 = 1e-1; end
if ~isfield(opts, 'beta_min'); opts.beta_min = 1e-6; end 
if ~isfield(opts, 'tol'); opts.tol = 1e-6; end


max_iter = opts.max_iter;
beta = opts.beta;
mu_0 = opts.mu_0;

tic
B = Bo; dist1 = zeros(max_iter,1); n = size(B,1);
for i = 1:max_iter
    if isfield(opts, 'Q')
        dist1(i) =  sum( abs( max(abs(B'*opts.Q),[],2) - ones(n,1) )  );
    end
    mu = mu_0*beta^(i);
    grad = Xtilde*sign(Xtilde'*B);
    gradB = grad'*B;
    grad = grad - 0.5*B*(gradB+ gradB');
    
    B_plus = B - mu*grad;
    [B,~] = qr(B_plus,0);
    err = norm(grad,'fro')/(1 + norm(B,'fro'));
    obj = sum(sum(abs(Xtilde'*B)));
    
    if obj < (1+1e-6)*opts.F_mialm 
        break
    end
    
end

ret.dist = dist1;
ret.err = err;
ret.iter = i;
ret.time = toc;
ret.obj = obj;

