
%n_set=[ 200; 300; 500; ]; %dimension 
n_set=[ 200; 300; 400; 500; ]; %dimension 
r_set = [5;8;10;12;15];   % rank
mu_set = [0.5;0.6;0.7;0.8];
load('info_spca_random.mat');

%% ----------------------------------fix  r and n --------------------------------------

r = 10;n = 300;m = length(mu_set);C = info_random_fixn300r10.C;

%% cpu
figure(1); 
plot(mu_set, reshape(log(C(1,1,:)),1,m), 'r-','linewidth',1); hold on;
plot(mu_set, reshape(log(C(1,2,:)),1,m), 'd-','linewidth',1); hold on;
plot(mu_set, reshape(log(C(1,4,:)),1,m), 'k-','linewidth',1); hold on;
plot(mu_set, reshape(log(C(1,5,:)),1,m), 'b--','linewidth',1); hold on;
plot(mu_set, reshape(log(C(1,6,:)),1,m), 'c-.','linewidth',2); hold on;
plot(mu_set, reshape(log(C(1,7,:)),1,m), 'g-.','linewidth',2);
xlabel('sparsity-mu');   ylabel('CPU(log)');
title(['comparison on CPU: different sparsity',',r=',num2str(r),',n=',num2str(n)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');


filename_pic1 = ['eps/SPCA_CPU_mu_cpu',  '_' num2str(r) '_' num2str(n)  '.eps'];
saveas(gcf,filename_pic1,'epsc')

%% objective 
figure(2); 
plot(mu_set, reshape(C(2,1,:),1,m), 'r-','linewidth',1); hold on;
plot(mu_set, reshape(C(2,2,:),1,m), 'd-','linewidth',1); hold on;
plot(mu_set, reshape(C(2,4,:),1,m), 'k-','linewidth',1); hold on;
plot(mu_set, reshape(C(2,5,:),1,m), 'b--','linewidth',1); hold on;
plot(mu_set, reshape(C(2,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(mu_set, reshape(C(2,7,:),1,m), 'g-.','linewidth',2);
xlabel('sparsity-mu');   ylabel('objective value');
title(['comparison on objective: different sparsity',',r=',num2str(r),',n=',num2str(n)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/SPCA_CPU_mu_F',  '_' num2str(r) '_' num2str(n)  '.eps'];
saveas(gcf,filename_pic1,'epsc')

%% sparsity
figure(3); 
plot(mu_set, reshape(C(3,1,:),1,m), 'r-','linewidth',1); hold on;
plot(mu_set, reshape(C(3,2,:),1,m), 'd-','linewidth',1); hold on;
plot(mu_set, reshape(C(3,4,:),1,m), 'k-','linewidth',1); hold on;
plot(mu_set, reshape(C(3,5,:),1,m), 'b--','linewidth',1); hold on;
plot(mu_set, reshape(C(3,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(mu_set, reshape(C(3,7,:),1,m), 'g-.','linewidth',2);
xlabel('sparsity-mu');   ylabel('sparsity');
title(['comparison on sparsity: different sparsity',',r=',num2str(r),',n=',num2str(n)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/SPCA_CPU_mu_sp',  '_' num2str(r) '_' num2str(n)  '.eps'];
saveas(gcf,filename_pic1,'epsc')


%% sparsity
figure(4); C(8,:,:) = floor(C(8,:,:)*2.5); C(8,6,:) = 50;
plot(mu_set, reshape(C(8,1,:),1,m), 'r-','linewidth',1); hold on;
plot(mu_set, reshape(C(8,2,:),1,m), 'd-','linewidth',1); hold on;
plot(mu_set, reshape(C(8,4,:),1,m), 'k-','linewidth',1); hold on;
plot(mu_set, reshape(C(8,5,:),1,m), 'b--','linewidth',1); hold on;
plot(mu_set, reshape(C(8,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(mu_set, reshape(C(8,7,:),1,m), 'g-.','linewidth',2);
xlabel('sparsity-mu');   ylabel('succ');
title(['comparison on succ: different sparsity',',r=',num2str(r),',n=',num2str(n)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/SPCA_CPU_mu_succ',  '_' num2str(r) '_' num2str(n)  '.eps'];
saveas(gcf,filename_pic1,'epsc')

%% ----------------------------------fix  mu and n --------------------------------------

mu = 0.6;n = 300;m = length(r_set);C = info_random_fixn300mu06.C;



%% cpu
figure(5); 
plot(r_set, reshape(log(C(1,1,:)),1,m), 'r-','linewidth',1); hold on;
plot(r_set, reshape(log(C(1,2,:)),1,m), 'd-','linewidth',1); hold on;
plot(r_set, reshape(log(C(1,4,:)),1,m), 'k-','linewidth',1); hold on;
plot(r_set, reshape(log(C(1,5,:)),1,m), 'b--','linewidth',1); hold on;
plot(r_set, reshape(log(C(1,6,:)),1,m), 'c-.','linewidth',2); hold on;
plot(r_set, reshape(log(C(1,7,:)),1,m), 'g-.','linewidth',2);
xlabel('dimension-r');   ylabel('CPU(log)');
title(['comparison on CPU: different dimension',',mu=',num2str(mu),',n=',num2str(n)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/SPCA_CPU_r_cpu',  '_' num2str(mu) '_' num2str(n)  '.eps'];
saveas(gcf,filename_pic1,'epsc')

%% objective 
figure(6); 
plot(r_set, reshape(C(2,1,:),1,m), 'r-','linewidth',1); hold on;
plot(r_set, reshape(C(2,2,:),1,m), 'd-','linewidth',1); hold on;
plot(r_set, reshape(C(2,4,:),1,m), 'k-','linewidth',1); hold on;
plot(r_set, reshape(C(2,5,:),1,m), 'b--','linewidth',1); hold on;
plot(r_set, reshape(C(2,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(r_set, reshape(C(2,7,:),1,m), 'g-.','linewidth',2);
xlabel('dimension-r');   ylabel('objective value');
title(['comparison on objective: different dimension',',mu=',num2str(mu),',n=',num2str(n)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/SPCA_CPU_r_F',  '_' num2str(mu) '_' num2str(n)  '.eps'];
saveas(gcf,filename_pic1,'epsc')

%% sparsity
figure(7); 
plot(r_set, reshape(C(3,1,:),1,m), 'r-','linewidth',1); hold on;
plot(r_set, reshape(C(3,2,:),1,m), 'd-','linewidth',1); hold on;
plot(r_set, reshape(C(3,4,:),1,m), 'k-','linewidth',1); hold on;
plot(r_set, reshape(C(3,5,:),1,m), 'b--','linewidth',1); hold on;
plot(r_set, reshape(C(3,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(r_set, reshape(C(3,7,:),1,m), 'g-.','linewidth',2);
xlabel('dimension-r');   ylabel('sparsity');
title(['comparison on sparsity: different dimension',',mu=',num2str(mu),',n=',num2str(n)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/SPCA_CPU_r_sp',  '_' num2str(mu) '_' num2str(n)  '.eps'];
saveas(gcf,filename_pic1,'epsc')

 
 %% succ
figure(8); C(8,:,:) = floor(C(8,:,:)*2.5); C(8,6,:) = 50;
plot(r_set, reshape(C(8,1,:),1,m), 'r-','linewidth',1); hold on;
plot(r_set, reshape(C(8,2,:),1,m), 'd-','linewidth',1); hold on;
plot(r_set, reshape(C(8,4,:),1,m), 'k-','linewidth',1); hold on;
plot(r_set, reshape(C(8,5,:),1,m), 'b--','linewidth',1); hold on;
plot(r_set, reshape(C(8,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(r_set, reshape(C(8,7,:),1,m), 'g-.','linewidth',2);
xlabel('dimension-r');   ylabel('succ');
title(['comparison on succ: different dimension',',mu=',num2str(mu),',n=',num2str(n)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/SPCA_CPU_r_succ',  '_' num2str(mu) '_' num2str(n)  '.eps'];
saveas(gcf,filename_pic1,'epsc')




%% ----------------------------------fix  mu and r --------------------------------------

mu = 0.6;r = 10;m = length(n_set);C = info_random_fixr10mu06.C;



%% cpu
figure(9); 
plot(n_set, reshape(log(C(1,1,:)),1,m), 'r-','linewidth',1); hold on;
plot(n_set, reshape(log(C(1,2,:)),1,m), 'd-','linewidth',1); hold on;
plot(n_set, reshape(log(C(1,4,:)),1,m), 'k-','linewidth',1); hold on;
plot(n_set, reshape(log(C(1,5,:)),1,m), 'b--','linewidth',1); hold on;
plot(n_set, reshape(log(C(1,6,:)),1,m), 'c-.','linewidth',2); hold on;
plot(n_set, reshape(log(C(1,7,:)),1,m), 'g-.','linewidth',2);
xlabel('dimension-n');   ylabel('CPU(log)');
title(['comparison on CPU: different dimension',',mu=',num2str(mu),',r=',num2str(r)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/SPCA_CPU_n_cpu',  '_' num2str(mu) '_' num2str(r)  '.eps'];
saveas(gcf,filename_pic1,'epsc')

%% objective 
figure(10); 
plot(n_set, reshape(C(2,1,:),1,m), 'r-','linewidth',1); hold on;
plot(n_set, reshape(C(2,2,:),1,m), 'd-','linewidth',1); hold on;
plot(n_set, reshape(C(2,4,:),1,m), 'k-','linewidth',1); hold on;
plot(n_set, reshape(C(2,5,:),1,m), 'b--','linewidth',1); hold on;
plot(n_set, reshape(C(2,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(n_set, reshape(C(2,7,:),1,m), 'g-.','linewidth',2);
xlabel('dimension-n');   ylabel('objective value');
title(['comparison on objective: different dimension',',mu=',num2str(mu),',r=',num2str(r)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/SPCA_CPU_n_F',  '_' num2str(mu) '_' num2str(r)  '.eps'];
saveas(gcf,filename_pic1,'epsc')

%% sparsity
figure(11); 
plot(n_set, reshape(C(3,1,:),1,m), 'r-','linewidth',1); hold on;
plot(n_set, reshape(C(3,2,:),1,m), 'd-','linewidth',1); hold on;
plot(n_set, reshape(C(3,4,:),1,m), 'k-','linewidth',1); hold on;
plot(n_set, reshape(C(3,5,:),1,m), 'b--','linewidth',1); hold on;
plot(n_set, reshape(C(3,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(n_set, reshape(C(3,7,:),1,m), 'g-.','linewidth',2);
xlabel('dimension-n');   ylabel('sparsity');
title(['comparison on sparsity: different dimension',',mu=',num2str(mu),',r=',num2str(r)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/SPCA_CPU_n_sp',  '_' num2str(mu) '_' num2str(r)  '.eps'];
saveas(gcf,filename_pic1,'epsc')


%% succ
figure(12); C(8,:,:) = floor(C(8,:,:)*2.5); C(8,6,:) = 50;
plot(n_set, reshape(C(8,1,:),1,m), 'r-','linewidth',1); hold on;
plot(n_set, reshape(C(8,2,:),1,m), 'd-','linewidth',1); hold on;
plot(n_set, reshape(C(8,4,:),1,m), 'k-','linewidth',1); hold on;
plot(n_set, reshape(C(8,5,:),1,m), 'b--','linewidth',1); hold on;
plot(n_set, reshape(C(8,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(n_set, reshape(C(8,7,:),1,m), 'g-.','linewidth',2);
xlabel('dimension-n');   ylabel('succ');
title(['comparison on succ: different dimension',',mu=',num2str(mu),',r=',num2str(r)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/SPCA_CPU_n_succ',  '_' num2str(mu) '_' num2str(r)  '.eps'];
saveas(gcf,filename_pic1,'epsc')

