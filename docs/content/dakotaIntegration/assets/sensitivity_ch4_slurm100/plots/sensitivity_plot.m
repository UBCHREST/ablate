clc;
clear all;
close all;
filename = '../dakota_tabular.dat';
opt = detectImportOptions(filename, 'filetype', 'text');
data = readtable(filename, opt);


theta1 = table2array(data(:,3)).*1E15;
theta2 = table2array(data(:,4)).*1E9;

f  = table2array(data(:,5));

sz = 10;

figure()
scatter(theta1,f,sz,'filled')

set(gca,'LineWidth',1,'FontSize',20,'FontWeight','normal')
set(get(gca,'XLabel'),'String','x_1','FontSize',25, 'fontname','times')
set(get(gca,'YLabel'),'String','Ignition Delay (Temperature)','FontSize',25, 'fontname','times')
set(gca,'ycolor','k')
saveas(gcf,'x1','png')
%--------------------------------------------------------------------------------------------
figure()
scatter(theta2,f,sz,'filled')

set(gca,'LineWidth',1,'FontSize',20,'FontWeight','normal')
set(get(gca,'XLabel'),'String','x_2','FontSize',25, 'fontname','times')
set(get(gca,'YLabel'),'String','Ignition Delay (Temperature)','FontSize',25, 'fontname','times')
set(gca,'ycolor','k')
saveas(gcf,'x2','png')



%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% total sensitivity index

S_theta = [  1.0733506007e+00  1.0645523590e+00
            1.0031489439e-02  7.3777144909e-03];

figure()
bar(S_theta)
ylim([0 1.1])
legend('first-order index', 'total effect index')
name = {'x_1';'x_2'};
set(gca,'LineWidth',1,'FontSize',20,'FontWeight','normal')
set(get(gca,'YLabel'),'String','Sobol indices','FontSize',25, 'fontname','times')
set(gca,'xticklabel',name)
saveas(gcf,'ST','png')

