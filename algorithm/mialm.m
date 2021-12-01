function [X,Z,ret]=mialm(A, manifold, f, h, opts)


%manifold inexact  solve nonsmooth manifold optimization problem 
% \min_x f(x) + h(A(x)), s.t. x\in manifold

% f: struct for function f, containing 
%    f.cost_grad: cost and gradient information of f
%    f.data: extra information of f
% h: struct for function f, containing 
%    h.cost: cost information of h
%    h.prox: proximal operator of h
%    h.prox: extra information of h
% A: struct for linear operator A
%    A.applyA: A(X); 
%    A.applyAT: AT(y). 
% manifold: struct for manifold. 
% 
%  Please cite the Manopt paper as well as the research paper:
%  @article{kangkang2019inexact,
%       title={An inexact augmented Lagrangian method for nonsmooth optimization on Riemannian manifold},
%       author={Kangkang, Deng and Zheng, Peng},
%       journal={arXiv preprint arXiv:1911.09900},
%       year={2019}
%  }

if ~isfield(opts, 'max_iter'); opts.max_iter = 1e3; end
if ~isfield(opts, 'rho'); opts.rho = 2; end
if ~isfield(opts, 'nu0'); opts.nu0 = 10; end
if ~isfield(opts, 'nu_max'); opts.nu_max = opts.nu0*1e3; end
if ~isfield(opts, 'nu_min'); opts.nu_min = 1e-2; end 
if ~isfield(opts, 'tol'); opts.tol = 1e-6; end
if ~isfield(opts, 'tau'); opts.tau = 0.9; end
if ~isfield(opts, 'verbosity'); opts.verbosity = 0; end
if ~isfield(opts, 'ALM_step'); opts.ALM_step = 1; end
if ~isfield(opts, 'sub_solver');opts.sub_solver = 3; end
if ~isfield(opts, 'gtol_ratio0'); opts.gtol_ratio0 = 1e0; end
if ~isfield(opts, 'record_file'); opts.record_file = ''; end
if ~isfield(opts, 'gtol0'); opts.gtol0 = 1e-1; end
if ~isfield(opts, 'maxitersub'); opts.maxitersub = 100; end
if ~isfield(opts, 'gtol_decrease'); opts.gtol_decrease = 0.8; end
if ~isfield(opts, 'extra_iter'); opts.extra_iter = 40; end


prefix = 'log';
tol = opts.tol;
rho = opts.rho;
sub_solver = opts.sub_solver;
deltak_cnt = 0;
opts.debug = 1;
%construct scale parameter

    
    


% construct Aop
if isnumeric(A)
    Aop = struct();
    Aop.applyA = @(X) A*X;
    Aop.applyAT = @(y) A'*y;
elseif isstruct(A) % A is struct, only validate its usability
    Aop = A;
    if ~isfield(A, 'applyA')
        error('A.applyA is not defined.');
    end
    if ~isfield(A, 'applyAT')
        error('A.applyAT is not defined.');
    end
else
    error('unsupported input of A.');
end


if ~isfield(f, 'cost_grad') 
    error('f is not defined.');
end

if ~isfield(f, 'data')
    f.data = {};
end



% check h (structure of function handle)

if isempty(h)
    h.cost = @(X,~) 0;
    h.prox = @(X,nuk,~) 0;
    h.is_empty = true;
else
    if ~isfield(h, 'prox') || ~isfield(h, 'cost')
        error('h is not defined.');
    end
    h.is_empty = true;
end
if ~isfield(h, 'data')
    h.data = {};
end



if ~isfield(opts, 'X0')
    X0 = manifold.rand();
else
    X0 = opts.X0;
end









nuk = opts.nu0;
gtol_ratio = opts.gtol_ratio0;


cstop = 0;
t = tic;


Y = Aop.applyA(X0);
Z = zeros(size(Y));
X = X0;

iter = 0;



if opts.verbosity > 0
    str0 = '     %6s';
    str2 = '      %6s';
    stra = ['\n%6s',str0,str2,str0,str0,str0, '    %4s', '    %3s', '  %6s','      %6s','    %6s'];
    str_head = sprintf(stra,...
        'iter','obj', 'deltak','kkt_X','kkt_Y','error', 'nuk', 'siter','snrmG', 'time', 'smsg');
    str_head_debug = sprintf('    %10s','  gtol_tgt');
   
    str_num = ['\n  %4d','  %+9.5e','  %+7.2e','  %+7.2e','  %+7.2e','  %7.2e','    %.1f','  %4d','      %8.2e','    %6.2f','    %-12s'];
    str_debug = ['%4.2e'];
    
    if ~isempty(opts.record_file)
        if ~exist(prefix, 'dir')
            mkdir(prefix);
        end
        record_fname = [prefix '/' opts.record_file];
        fid = fopen(record_fname, 'w');
    else
        fid = 1;
    end
end




%ftol_inc_step = 0;


RGBB_extra_iter = 0;

AXy = Aop.applyA(X) - Y;
deltak = norm(AXy);




out.nrmG = 1;
sub_iter = 0;


gtol_bnd = opts.gtol0;
maxitersub = opts.maxitersub;

% initialize output
ret = struct();
ret.flag = 99;
ret.msg = 'exceed max iteration';
time_arr = zeros(opts.max_iter,1);
obj_arr = zeros(opts.max_iter,1);
error_arr = zeros(opts.max_iter,3);
[fcost,~] = f.cost_grad(X, f.data{:});
obj_arr(1) = fcost + h.cost(Aop.applyA(X),h.data{:});
while iter<opts.max_iter && ~cstop
    iter = iter+1;
    X0 = X;
    ALM_step = opts.ALM_step;
    
  
    % set gtol
    gtol_min = 1e-6;
    
    
    gtol = max([gtol_ratio*1e-6 * sqrt(deltak), gtol_bnd, gtol_min]);
    
    
    
    switch sub_solver
        case 2
            problem.M = manifold;
            options_manopt.verbosity = 0;
            options_manopt.maxiter = 100;
            options_manopt.tolgradnorm = gtol;
            [X, ~, out] = trustregions(problem, X0, options_manopt);
        case 3
           optRGB = opts;
            
           optRGB.xtol = 1e-5;  optRGB.ftol = 1e-8;  optRGB.gtol = gtol;
           optRGB.alpha = 1e-3; optRGB.rhols = 1e-6; optRGB.gamma = 0.85;
           optRGB.nt = 5;       optRGB.eta = 0.2;    optRGB.STPEPS = 1e-10;
           optRGB.maxit = maxitersub + RGBB_extra_iter;
           optRGB.record = opts.verbosity > 1;
           if optRGB.record
               optRGB.record_fid = fid;
           end
           [X, ~, out] = RGBB(X0, @fun_ARNT, manifold, optRGB, Aop, f, h, Z, nuk);
           
           if out.iter == optRGB.maxit
               RGBB_extra_iter = min(opts.extra_iter, max(2 * RGBB_extra_iter, 10));
           else
               gtol_bnd = gtol_bnd * opts.gtol_decrease;
           end
           sub_iter = sub_iter + out.iter;
           
    end
    
    acc_time = toc(t);
    time_arr(iter+1) = acc_time;
    
    
    
    AX = Aop.applyA(X); AXZ = AX  - Z/nuk;
    Y = h.prox(AXZ, 1/nuk, h.data{:});
    deltak_p =deltak;
    deltak = norm(AX - Y,'fro')/(1+norm(AX,'fro') + norm(Y,'fro'));
    
    deltak_ratio = deltak/deltak_p;
    
    
    % multiplier update: Z
    Z = Z-ALM_step*nuk*(AX - Y);
    
    [fcost,fgrad] = f.cost_grad(X, f.data{:});
    hprox = h.prox( AX - Z, 1, h.data{:});
    kkt_X = norm(manifold.proj(X, fgrad - Aop.applyAT(Z)), 'fro')/(1 + norm(fgrad,'fro'));
    kkt_Y = norm(AX - hprox, 'fro')/(1+norm(AX,'fro'));
    kkt_error = max(kkt_X, kkt_Y);
   
    obj = fcost + h.cost(Aop.applyA(X),h.data{:});
    obj_arr(iter+1) = obj;
    error_arr(iter,:) = [deltak,kkt_X,kkt_Y];
    % adjust sigmak such that deltak & etaK2 decrease at the same rate
    sigtol = opts.tau;
    if deltak_ratio > sigtol
        deltak_cnt = deltak_cnt + 1;
    else
        deltak_cnt = 0;
    end

    
    if deltak_ratio > sigtol && deltak >= tol
        nuk = min(rho * nuk, opts.nu_max);
    elseif deltak < tol && kkt_error > tol*10
        nuk = max(nuk / rho, opts.nu_min);
    end
   
    cstop = (kkt_error<tol  && deltak < tol );
    if cstop
        ret.flag = 0;
        ret.msg = 'converge';
    end
    
    % print
    if opts.verbosity > 0
        % print header
        if iter == 1 || opts.verbosity > 1
            fprintf(fid, str_head);
            if opts.debug
                fprintf(fid, str_head_debug);
            end
        end
        
        % print iteration info
        switch sub_solver
            case 2
                fprintf(fid, str_num,iter,deltak,kkt_error, nuk, out(end).iter, out(end).gradnorm, acc_time, '');
                if opts.debug
                    fprintf(fid, str_debug, gtol);
                end
            case 3
                fprintf(fid, str_num,iter,obj,deltak,kkt_X,kkt_Y,kkt_error, nuk, out.iter,out.nrmG, acc_time, out.msg);
                if opts.debug
                    fprintf(fid, str_debug, gtol);
                end
        end
        
    end
    
  
 
    
end

sub_iter = sub_iter/iter;



tsolve_ALM = toc(t);
if iter < opts.max_iter
    ret.flag = 1;
else
    ret.flag = 0;
end
ret.obj_arr = obj_arr;
ret.time_arr = time_arr;
ret.error_arr = error_arr;
ret.time = tsolve_ALM;
ret.iter = iter;
ret.deltak = deltak;
ret.X = X;
ret.Z = Z;
ret.Y = Y;
ret.nu = nuk;
ret.obj = f.cost_grad(X,f.data{:}) + h.cost(Aop.applyA(X),h.data{:});
ret.sub_iter = sub_iter;
ret.nrmG = out.nrmG;
ret.etaD = kkt_X;
ret.etaC = kkt_Y;
[n,k] = size(AX);
ret.sparsity = sum(sum(abs(AX)<=1e-4))/(n*k);

if opts.verbosity
    hrule = repmat('-', 1, 80);
    fprintf(fid, '\n%s\n', hrule);
    fprintf(fid, '- MIALM OUTPUT\n');
    fprintf(fid, [hrule, '\n']);
    fprintf(fid, '  exit code = %d (%s)\n', ret.flag, ret.msg);
    fprintf(fid, [hrule, '\n']);
    fprintf(fid, '  iter = %d\n', iter);
    fprintf(fid, '  time = %.2f\n', ret.time);
    fprintf(fid, '  obj = %.12e \n', ret.obj);
    fprintf(fid, '  feasibility:\n');
    fprintf(fid, '    etaP = %8.2e    etaD  = %8.2e     etaC  = %8.2e\n', deltak, kkt_X, kkt_Y);
    fprintf(fid, [hrule, '\n']);
end


if opts.verbosity > 0 && ~isempty(opts.record_file)
    fclose(fid);
end





function [f,g] = fun_ARNT(X, Aop, f, h, Z,nuk)

% apply A
AXZ = Aop.applyA(X) - Z/nuk;
AXZ_prox = h.prox(AXZ,1/nuk,h.data{:});
% apply AT (handle)
AAXZ = Aop.applyAT(AXZ - AXZ_prox);



[f1,g1] = f.cost_grad(X,f.data{:});


g = g1 + nuk*AAXZ;

f = f1 + h.cost(AXZ_prox,h.data{:}) + nuk/2*norm(AXZ - AXZ_prox, 'fro')^2;

end



end



