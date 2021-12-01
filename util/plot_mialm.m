function plot_mialm(mialm_time,mialm_obj,mialm_error, madmm_time,madmm_obj,madmm_error,iter1,iter2)

colors  = ['m' 'b' 'r' 'c' 'k' 'g' 'y'];
lines   = [':' '-' '-.' '--','-','--',':'];
markers = ['x' '*' 's' 'd' 'v' '^' 'yo'];
lines   = {'k-' 'b--' 'r-.' 'c:' 'g-*' 'm-' 'r-'};

fig = figure(1);
clf;
plot(mialm_time(1:iter1),mialm_obj(1:iter1),lines{1},'LineWidth',2,'MarkerSize',3);
hold on;
plot(madmm_time(1:iter2),madmm_obj(1:iter2),lines{2},'LineWidth',2,'MarkerSize',3)
legend( {'MIALM','MADMM'}, 'location','northeast');
axis( [0 8 0 20] );
xlabel('cpu time(s)');
ylabel('objective value');
%title('The objective value compare');
set(gca,'fontsize',16);
print(fig, '-depsc2' , 'cms_cpu_obj_cmp_.eps');
fig = figure(2);
clf
for kkk = 1:3
     plot(mialm_time(1:iter1),log(mialm_error(1:iter1,kkk)),lines{1+(kkk-1)*2},'LineWidth',2,'MarkerSize',3);
    hold on;
end
for kkk = 1:3
     plot(madmm_time(1:iter2),log(madmm_error(1:iter2,kkk)),lines{2+(kkk-1)*2},'LineWidth',2,'MarkerSize',3);
    hold on;
end
% plot(time_arr_Rsub(1:maxit_att_Rsub),obj_arr_Rsub(1:maxit_att_Rsub))
% legend([line(1), line(3),line(5)],{'MIALM-\eta_p', 'MIALM-\eta_d','MIALM-\eta_C'},...
%         'Location', 'northeast');
% axesNew = axes('position',get(gca,'position'),'visible','off');
% 
% legend(axesNew,[line(2), line(4),line(6)],{'MADMM-\eta_p', 'MADMM-\eta_d','MADMM-\eta_C'},...
%         'Location', 'southeast');
legend( {'MIALM-\eta_p','MIALM-\eta_d','MIALM-\eta_C','MADMM-\eta_p','MADMM-\eta_d','MADMM-\eta_C'}, 'location','northeast','NumColumns',2);
axis( [0 8 -15 5] );
xlabel('cpu time (s)');
ylabel('error (log)');
%title('The objective value compare');
set(gca,'fontsize',16);
print(fig, '-depsc2' , 'cms_cpu_error_cmp_.eps');