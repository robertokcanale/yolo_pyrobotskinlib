palm = readtable('data_files/palm.txt');
thumb = readtable('data_files/thumb.txt');
index = readtable('data_files/index.txt');
middle = readtable('data_files/middle.txt');
ring = readtable('data_files/ring.txt');
pinkie = readtable('data_files/pinkie.txt');
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