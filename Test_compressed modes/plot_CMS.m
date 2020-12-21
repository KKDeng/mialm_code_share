
n_set = [128;256;400;512;600;700;800];
r_set = [2;4;6;8;10];   % rankr_set = [4;5;6;7;8];   % rank
mu_set = [0.05;0.1;0.15;0.2;0.25;0.3];
load('info_CMS.mat');

%% ----------------------------------fix  r and n --------------------------------------

r = 4;n = 256;m = length(mu_set);C = new_info_CMs_fixr4n256.C;

%% cpu
figure(1); 
plot(mu_set, reshape(C(1,1,:),1,m), 'r-','linewidth',1); hold on;
plot(mu_set, reshape(C(1,2,:),1,m), 'd-','linewidth',1); hold on;
plot(mu_set, reshape(C(1,4,:),1,m), 'k-','linewidth',1); hold on;
plot(mu_set, reshape(C(1,5,:),1,m), 'b--','linewidth',1); hold on;
plot(mu_set, reshape(C(1,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(mu_set, reshape(C(1,7,:),1,m), 'g-.','linewidth',2);
xlabel('sparsity-mu');   ylabel('CPU');
title(['comparison on CPU: different sparsity',',r=',num2str(r),',n=',num2str(n)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/CMS_CPU_mu_cpu',  '_' num2str(r) '_' num2str(n)  '.eps'];
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

filename_pic1 = ['eps/CMS_CPU_mu_F',  '_' num2str(r) '_' num2str(n)  '.eps'];
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

filename_pic1 = ['eps/CMS_CPU_mu_sp',  '_' num2str(r) '_' num2str(n)  '.eps'];
saveas(gcf,filename_pic1,'epsc')


%% succ
figure(4); C(4,6,:) = 50;
plot(mu_set, reshape(C(4,1,:),1,m), 'r-','linewidth',1); hold on;
plot(mu_set, reshape(C(4,2,:),1,m), 'd-','linewidth',1); hold on;
plot(mu_set, reshape(C(4,4,:),1,m), 'k-','linewidth',1); hold on;
plot(mu_set, reshape(C(4,5,:),1,m), 'b--','linewidth',1); hold on;
plot(mu_set, reshape(C(4,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(mu_set, reshape(C(4,7,:),1,m), 'g-.','linewidth',2);
xlabel('sparsity-mu');   ylabel('succ');
title(['comparison on succ: different sparsity',',r=',num2str(r),',n=',num2str(n)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/CMS_CPU_mu_succ',  '_' num2str(r) '_' num2str(n)  '.eps'];
saveas(gcf,filename_pic1,'epsc')

%% ----------------------------------fix  mu and n --------------------------------------

mu = 0.1;n = 256;m = length(r_set);C = new_info_CMs_fixmu01n256.C;



%% cpu
figure(5); 
plot(r_set, reshape(C(1,1,:),1,m), 'r-','linewidth',1); hold on;
plot(r_set, reshape(C(1,2,:),1,m), 'd-','linewidth',1); hold on;
plot(r_set, reshape(C(1,4,:),1,m), 'k-','linewidth',1); hold on;
plot(r_set, reshape(C(1,5,:),1,m), 'b--','linewidth',1); hold on;
plot(r_set, reshape(C(1,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(r_set, reshape(C(1,7,:),1,m), 'g-.','linewidth',2);
xlabel('dimension-r');   ylabel('CPU');
title(['comparison on CPU: different dimension',',mu=',num2str(mu),',n=',num2str(n)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/CMS_CPU_r_cpu',  '_' num2str(mu) '_' num2str(n)  '.eps'];
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

filename_pic1 = ['eps/CMS_CPU_r_F',  '_' num2str(mu) '_' num2str(n)  '.eps'];
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

filename_pic1 = ['eps/CMS_CPU_r_sp',  '_' num2str(mu) '_' num2str(n)  '.eps'];
saveas(gcf,filename_pic1,'epsc')


%% sparsity
figure(8); C(4,6,:) = 50;
plot(r_set, reshape(C(4,1,:),1,m), 'r-','linewidth',1); hold on;
plot(r_set, reshape(C(4,2,:),1,m), 'd-','linewidth',1); hold on;
plot(r_set, reshape(C(4,4,:),1,m), 'k-','linewidth',1); hold on;
plot(r_set, reshape(C(4,5,:),1,m), 'b--','linewidth',1); hold on;
plot(r_set, reshape(C(4,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(r_set, reshape(C(4,7,:),1,m), 'g-.','linewidth',2);
xlabel('dimension-r');   ylabel('succ');
title(['comparison on succ: different dimension',',mu=',num2str(mu),',n=',num2str(n)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/CMS_CPU_r_succ',  '_' num2str(mu) '_' num2str(n)  '.eps'];
saveas(gcf,filename_pic1,'epsc')


%% ----------------------------------fix  mu and r --------------------------------------

mu = 0.1;r = 4;m = length(n_set);C = new_info_CMs_fixmu01r4.C;



%% cpu
figure(9); 
plot(n_set, reshape(C(1,1,:),1,m), 'r-','linewidth',1); hold on;
plot(n_set, reshape(C(1,2,:),1,m), 'd-','linewidth',1); hold on;
plot(n_set, reshape(C(1,4,:),1,m), 'k-','linewidth',1); hold on;
plot(n_set, reshape(C(1,5,:),1,m), 'b--','linewidth',1); hold on;
plot(n_set, reshape(C(1,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(n_set, reshape(C(1,7,:),1,m), 'g-.','linewidth',2);
xlabel('dimension-n');   ylabel('CPU');
title(['comparison on CPU: different dimension',',mu=',num2str(mu),',r=',num2str(r)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/CMS_CPU_n_cpu',  '_' num2str(mu) '_' num2str(r)  '.eps'];
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

filename_pic1 = ['eps/CMS_CPU_n_F',  '_' num2str(mu) '_' num2str(r)  '.eps'];
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

filename_pic1 = ['eps/CMS_CPU_n_sp',  '_' num2str(mu) '_' num2str(r)  '.eps'];
saveas(gcf,filename_pic1,'epsc')


figure(12); C(4,6,:) = 50;
plot(n_set, reshape(C(4,1,:),1,m), 'r-','linewidth',1); hold on;
plot(n_set, reshape(C(4,2,:),1,m), 'd-','linewidth',1); hold on;
plot(n_set, reshape(C(4,4,:),1,m), 'k-','linewidth',1); hold on;
plot(n_set, reshape(C(4,5,:),1,m), 'b--','linewidth',1); hold on;
plot(n_set, reshape(C(4,6,:),1,m), 'c-.','linewidth',2); hold on;
plot(n_set, reshape(C(4,7,:),1,m), 'g-.','linewidth',2);
xlabel('dimension-n');   ylabel('succ');
title(['comparison on succ: different dimension',',mu=',num2str(mu),',r=',num2str(r)]);
legend('ManPG','ManPG-adap','SOC','PAMAL','MIALM','MADMM');

filename_pic1 = ['eps/CMS_CPU_n_succ',  '_' num2str(mu) '_' num2str(r)  '.eps'];
saveas(gcf,filename_pic1,'epsc')
