%Basic matrix operations and comptations

U1 = [1; 2; 3], U2 = [4, 5, 6] % chain mutliple commands in single line (use space or colon)
u1 = 5; u2 = 6; % chain and suppress print

V = [1;2;3;4]
V.^2
2.^V

A = [1 2 3; 4 5 6; 7 8 9]
B = [1 2; 3 4; 5 6]
A * B
A .* B

1 ./ B
log(B)
exp(B)
abs(B)
ones(length(B),1) + B
1 + B
[B(:,1) + 1, B(:,2)]
[B(:,1) + 1, B(:,2:size(B,2))]

a = [1 15.7 0.05 2]
b = [a;a;a]
ceil(b)
floor(b)
b = [b; 1.3 5.6 90.9 09.0]
max(b)
min(b)
b < 3
find(b < 3)

M = magic(5)
[r,c] = find(M >= 7)
M(r(1),c(1))

sum(M)
sum(M,1) % col-wise sum
sum(M,2) % row-wise sum

M2 = M .* eye(3)
sum(sum(M2)) % sum the diagonal using eye, TL to BR
M3 = M .* flip(eye(3))
sum(sum(M3)) % sum the diagonal using eye flipped, BL to TR

prod(M)
max(M,[],1) % col-wise max
max(M,[],2) # row-wise max
max(max(M))
max(M(:))

M * pinv(M)
abs(round(M * pinv(M))) == eye(3) % the identity matrix is matrix by its inverse for same dimensions

R = rand(3)

% Matrix plotting
pwd
cd 'C:\Users\Admin\Documents\Coursera\Machine_Learning'
pwd

t = [0: 0.01 : 0.98]
y1 = sin(2 * pi * 4 * t)
figure(1)
plot(t,y1)
hold on % allow additional plots to current plot
y2 = cos(2 * pi * 4 * t)
xlabel('Time')
ylabel('Value')
legend('sin','cos')
title('Sin vs Cos plot')
print -dpng 'DemoPlot1.png'
hold off
close(1) % close plot figure 1
figure(2)
plot(y1,y2)
clf % clear figure
figure(3)
subplot(1,2,1) % divide figure 3 in to 1 * 2 grid and access first element
plot(t,y1)
axis([0 0.5 -0.5 0.5]) % set axis limits [x-low x-high y-low y-high]
subplot(1,2,2) # access second element of figure 3
plot(t,y2)
axis([0.2 1 -1 0.5])
get(0,"currentfigure") % return number of current figure

M9 = magic(9)
figure(4)
imagesc(M9), colorbar, colormap gray; % create heatmap and print color bar
colormap(cool);

% Control statements
o = zeros(10,1)
for i=1:length(o),
  o(i) = 2 ^ (i * 2);
end
disp(o)

i = 1
while true,
  o(i) = 125;
  i = i+1;
  if i == 6,
    break
  elseif i <= 3,
    continue
  else,
    disp('at iteration 4 or 5')  
  end  
end
disp(o)
 
% Using Functions
%eg - create file in current working diectory (pwd)
%eg - function s = squareMe(x)
%eg - s = x^2;
%eg - to use: n = squareMe(5)

addpath('C:\Users\Admin\Documents\Coursera\Machine_Learning\Functions') % add functions to path

%eg - function [s1,s2] = squareAndCubeMe(x)
%eg - s1 = x^2;
%eg - s2 = x^3;
[y1,y2] = squareAndCubeMe(5)

% cost function example
X = [1 1; 1 2; 1 3;]; % training examples
Y = [2; 3; 4]; % actuals
r2 = [];
for i=1:5,
theta = [1; 0.4+i/5]; % feature parameters / prediction coefficients
c = costFunction(X,Y,theta);
r = [theta; c];
r2 = [r2, r];
end
disp(r2)

keyboard; % this command allows debugging during program execution (ie: inspect variables, hand-enter some commands)
return; % return from debug mode to normal mode

clear, close all, clc;

