function saveEps(figure,filename,width,height)
% Save a figure to EPS
% pic = ['../rapport/Q1/' 'Q1'] ;
% h = figure ;
% saveEps(h,pic,13,10) ;
set(figure,'InvertHardcopy','on');
set(figure,'PaperUnits', 'centimeters');
papersize = get(figure, 'PaperSize');
left = (papersize(1)- width)/2;
bottom = (papersize(2)- height)/2;
myfiguresize = [left, bottom, width, height];
set(figure,'PaperPosition', myfiguresize);
print(filename,'-depsc');

end