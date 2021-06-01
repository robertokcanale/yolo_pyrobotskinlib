id = '_integral_moments';
palm = readtable(strcat('data_files/palm', id, '.txt'));
thumb = readtable(strcat('data_files/thumb', id, '.txt'));
index = readtable(strcat('data_files/index', id, '.txt'));
middle = readtable(strcat('data_files/middle', id, '.txt'));
ring = readtable(strcat('data_files/ring', id, '.txt'));
pinkie = readtable(strcat('data_files/pinkie', id, '.txt'));
set(0,'DefaultTextFontname', 'CMU Serif')

dg = '#297C00';
dark_green = sscanf(dg(2:end),'%2x%2x%2x',[1 3])/255;
br = '#90603C';
brown = sscanf(br(2:end),'%2x%2x%2x',[1 3])/255;
gr = '#ECE8E5';
gray = sscanf(gr(2:end),'%2x%2x%2x',[1 3])/255;

x = 1:length(palm.Var1);

% Tile 1
subplot(1,3,1)
plot(x,palm.Var2,'b','LineWidth',2');
hold on

plot(x, thumb.Var2,'Color',' #297C00 ','LineWidth',2) 
hold on

plot(x,index.Var2,'r','LineWidth',2)
hold on

plot(x,middle.Var2,'g', 'LineWidth',2)
hold on

plot(x,ring.Var2,'Color','#90603C','LineWidth',2)
hold on

plot(x,pinkie.Var2,'Color','#ECE8E5','LineWidth',2)
hold on
axis on

xlabel('Time ms','FontSize',20,'Interpreter', 'latex') 
ylabel('Intensity','FontSize',20,'Interpreter', 'latex') 
title('X','FontSize',20,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

% Tile 2
subplot(1,3,2)

plot(x,palm.Var3,'b','LineWidth',2);
hold on

plot(x, thumb.Var3,'Color',' #297C00 ','LineWidth',2) 
hold on

plot(x,index.Var3,'r','LineWidth',2)
hold on

plot(x,middle.Var3,'g','LineWidth',2)
hold on

plot(x,ring.Var3,'Color','#90603C','LineWidth',2)
hold on

plot(x,pinkie.Var3,'Color','#ECE8E5','LineWidth',2)
hold on
axis on

xlabel('Time ms','FontSize',20,'Interpreter', 'latex') 
ylabel('Intensity','FontSize',20,'Interpreter', 'latex') 
title('Y','FontSize',20,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

% Tile 3
subplot(1,3,3)

plot(x,palm.Var4,'b','LineWidth',2);
hold on

plot(x, thumb.Var4, 'Color','#297C00' ,'LineWidth',2) 
hold on

plot(x,index.Var4,'r','LineWidth',2)
hold on

plot(x,middle.Var4,'g','LineWidth',2)
hold on

plot(x,ring.Var4,'Color','#90603C', 'LineWidth',2)
hold on

plot(x,pinkie.Var4,'Color','#ECE8E5','LineWidth',2)
hold on
axis on

xlabel('Time ms','FontSize',20,'Interpreter', 'latex') 
ylabel('Intensity','FontSize',20,'Interpreter', 'latex') 
title('Z','FontSize',20,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');

sgtitle('Finger Moments Integral','FontSize',30,'FontWeight','bold', 'FontName', 'CMU Serif', 'Interpreter', 'latex');


% Construct a Legend with the data from the sub-plots
hL = legend({'palm','thumb', 'index', 'middle', 'ring', 'pinkie'}, 'FontSize',15,'Interpreter', 'latex');
% Programatically move the Legend
newPosition = [0.75 0.75 0.3 0.3];
newUnits = 'normalized';
set(hL,'Position', newPosition,'Units', newUnits);
