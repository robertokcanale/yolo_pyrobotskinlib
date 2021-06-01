force = readtable(strcat('data_files/total_force.txt'));
moment = readtable(strcat('data_files/total_moment.txt'));

set(0,'DefaultTextFontname', 'CMU Serif')
x = 1:length(force.Var1);

% Tile 1
subplot(2,3,1)
plot(x,force.Var2, 'r','LineWidth',2);
hold on
axis on

xlabel('Time ms','FontSize',20,'Interpreter', 'latex') 
ylabel('Intensity','FontSize',20,'Interpreter', 'latex') 
title('Force X','FontSize',20,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

% Tile 2
subplot(2,3,2)

plot(x,force.Var3, 'g','LineWidth',2);
hold on

hold on
axis on

xlabel('Time ms','FontSize',20,'Interpreter', 'latex') 
ylabel('Intensity','FontSize',20,'Interpreter', 'latex') 
title('Force Y','FontSize',20,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

% Tile 3
subplot(2,3,3)

plot(x,force.Var4,'b','LineWidth',2);
hold on
hold on
axis on

xlabel('Time ms','FontSize',20,'Interpreter', 'latex') 
ylabel('Intensity','FontSize',20,'Interpreter', 'latex') 
title('Force Z','FontSize',20,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

% Tile 4
subplot(2,3,4)
plot(x,moment.Var2,'r','LineWidth',2);
hold on
axis on

xlabel('Time ms','FontSize',20,'Interpreter', 'latex') 
ylabel('Intensity','FontSize',20,'Interpreter', 'latex') 
title('Moment X','FontSize',20,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

% Tile 5
subplot(2,3,5)

plot(x,moment.Var3,'g', 'LineWidth',2);
hold on

hold on
axis on

xlabel('Time ms','FontSize',20,'Interpreter', 'latex') 
ylabel('Intensity','FontSize',20,'Interpreter', 'latex') 
title('Moment Y','FontSize',20,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

% Tile 6
subplot(2,3,6)

plot(x,moment.Var4,'b','LineWidth',2);
hold on
hold on
axis on

xlabel('Time ms','FontSize',20,'Interpreter', 'latex') 
ylabel('Intensity','FontSize',20,'Interpreter', 'latex') 
title('Moment Z','FontSize',20,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

sgtitle('Total Hand Force and Moment Integral','FontSize',30,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');



