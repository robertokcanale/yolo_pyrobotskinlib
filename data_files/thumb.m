force = readtable(strcat('thumb_integral_forces.txt'));
moment = readtable(strcat('thumb_integral_moments.txt'));

set(0,'DefaultTextFontname', 'CMU Serif')
x = 1:length(force.Var1);

% Tile 1
subplot(2,3,1)
plot(x,force.Var2, 'r','LineWidth',1);
hold on
axis on
xlabel('Samples','FontSize',12,'FontName', 'CMU Serif','Interpreter', 'latex') 
ylabel('Intensity','FontSize',12,'FontName', 'CMU Serif','Interpreter', 'latex') 
title('Force X','FontSize',20,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

% Tile 2
subplot(2,3,2)

plot(x,force.Var3, 'g','LineWidth',1);
hold on

hold on
axis on

xlabel('Samples','FontSize',12,'FontName', 'CMU Serif','Interpreter', 'latex') 
ylabel('Intensity','FontSize',12,'FontName', 'CMU Serif','Interpreter', 'latex') 
title('Force Y','FontSize',18,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

% Tile 3
subplot(2,3,3)

plot(x,force.Var4,'b','LineWidth',1);
hold on
hold on
axis on

xlabel('Samples','FontSize',12,'FontName', 'CMU Serif','Interpreter', 'latex') 
ylabel('Intensity','FontSize',12,'FontName', 'CMU Serif','Interpreter', 'latex') 
title('Force Z','FontSize',18,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

% Tile 4
subplot(2,3,4)
plot(x,moment.Var2,'r','LineWidth',1);
hold on
axis on

xlabel('Samples','FontSize',12,'FontName', 'CMU Serif','Interpreter', 'latex') 
ylabel('Intensity','FontSize',12,'FontName', 'CMU Serif','Interpreter', 'latex') 
title('Moment X','FontSize',18,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

% Tile 5
subplot(2,3,5)

plot(x,moment.Var3,'g', 'LineWidth',1);
hold on

hold on
axis on

xlabel('Samples','FontSize',12,'FontName', 'CMU Serif','Interpreter', 'latex') 
ylabel('Intensity','FontSize',12,'FontName', 'CMU Serif','Interpreter', 'latex') 
title('Moment Y','FontSize',18,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

% Tile 6
subplot(2,3,6)

plot(x,moment.Var4,'b','LineWidth',1);
hold on
hold on
axis on

xlabel('Samples','FontSize',12,'FontName', 'CMU Serif','Interpreter', 'latex') 
ylabel('Intensity','FontSize',12,'FontName', 'CMU Serif','Interpreter', 'latex') 
title('Moment Z','FontSize',18,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

sgtitle('Total Thumb Force and Moment Integral','FontSize',25,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');



