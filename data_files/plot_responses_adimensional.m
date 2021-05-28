id = '_integral_forces';
palm = readtable(strcat('data_files/palm', id, '.txt'));
thumb = readtable(strcat('data_files/thumb', id, '.txt'));
index = readtable(strcat('data_files/index', id, '.txt'));
middle = readtable(strcat('data_files/middle', id, '.txt'));
ring = readtable(strcat('data_files/ring', id, '.txt'));
pinkie = readtable(strcat('data_files/pinkie', id, '.txt'));
set(0,'DefaultTextFontname', 'CMU Serif')
x = 1:length(palm.Var1);

plot(x,palm.Var2);
hold on

plot(x, thumb.Var2) 
hold on

plot(x,index.Var2)
hold on

plot(x,middle.Var2)
hold on

plot(x,ring.Var2)
hold on

plot(x,pinkie.Var2)
hold on

legend({'palm','thumb', 'index', 'middle', 'ring', 'pinkie'}, 'FontSize',15,'Interpreter', 'latex')
title('Hand Response','FontSize',30,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');
xlabel('Time ms','FontSize',20,'Interpreter', 'latex') 
ylabel('Average Response','FontSize',20,'Interpreter', 'latex') 
