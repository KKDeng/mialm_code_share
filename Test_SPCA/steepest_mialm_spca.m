function [x, info] = steepest_mialm_spca(problem, x, options)

    
    
    
    
    
   
    
    % If no initial point x is given by the user, generate one at random.
   
    
    AtA = problem.AtA;
    
    % Compute objective-related quantities for x.
    if(options.flag==1)
        BX = options.BX;
        egrad = problem.egrad(x,BX);
        xgx = x'*egrad;
        grad = egrad - 0.5*x*(xgx+xgx');
        gradnorm = norm(grad,'fro');
        oldF = problem.cost(x,BX);
    else
        BX = AtA*x;
        egrad = problem.egrad(x,BX);
        xgx = x'*egrad;
        grad = egrad - 0.5*x*(xgx+xgx');
        gradnorm = norm(grad,'fro');
        oldF = problem.cost(x,BX);
    end
    
    
    gama = 0.001;
    iter = 0;
    num_linesearch = 0;
    F = zeros(options.maxiter+1,0);
    
    stepsize = options.stepsize;
    
    
    timetic = tic();
    temp = zeros(100000,1);
    % Start iterating until stopping criterion triggers.
    while (gradnorm>options.tolgradnorm && iter<options.maxiter) || iter==0 
        
        
        oldx = x;
         
        x = oldx - stepsize*grad;
        [U, SIGMA, S] = svd(x'*x);   SIGMA =diag(SIGMA);    newx = x*(U*diag(sqrt(1./SIGMA))*S');
        
       
        
       F(iter+1) = oldF;
       %maxF = max(F(1:min(4,iter+1)));
        
       BX = AtA*newx;        
       newF = problem.cost(newx,BX);t = 1;
         
        while(newF>oldF - gama/(iter+1)*stepsize*gradnorm^2 && t<5)
            stepsize = stepsize*0.5;
            x = oldx - stepsize*grad;
            [U, SIGMA, S] = svd(x'*x);   SIGMA =diag(SIGMA);    newx = x*(U*diag(sqrt(1./SIGMA))*S');
            BX = AtA*newx;
            newF = problem.cost(newx,BX);       
            t = t+1;
            num_linesearch = num_linesearch + 1;
        end
        
        egrad = problem.egrad(newx,BX);
        xgx = newx'*egrad;
        newgrad = egrad - 0.5*newx*(xgx+xgx');
        %newgradnorm = problem.M.norm(newx, newgrad);
        newgradnorm = norm(newgrad,'fro');
        
         grad_transp = problem.M.transp(oldx, newx, grad);
        y = newgrad - grad_transp;
        s = -stepsize*grad_transp; 
        
%         y = newgrad - grad;
%         s = newx - oldx;
        
        
        stepsize = abs(norm(s,'fro')^2/sum(sum(s.*y)))*1;
      %  stepsize = min(stepsize,abs(sum(sum(s.*y))/sum(sum(y.*y))));
        
        x = newx;
        grad = newgrad;
        gradnorm = newgradnorm;
        oldF = newF;
        
        
        
   
        
        
        % iter is the number of iterations we have accomplished.
        iter = iter + 1;
        

        temp(iter) = gradnorm;
        
    end
    
    times = toc(timetic);
    
    info(1).time = times;
    info(1).iter = iter;
    info(1).gradnorm = gradnorm;
    info(1).stepsize = stepsize;
    info(1).BX = BX;
    info(1).num_linesearch = num_linesearch;
    
    
    

 
    
    
   
    
    
end
