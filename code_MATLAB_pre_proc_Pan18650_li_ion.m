%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                       %
%  Alexandre B. de Lima                                 %
%                                                       %
%   02/09/2020                                          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Given a drive cycle applied on the ?2.9Ah Panasonic 18650PF battery
% Dataset: Panasonic 18650PF Li-ion Battery
% https://www.mendeley.com/profiles/phillip-kollmeyer/
 
clear all 
close all

% =========================================================================
% INPUT DATA

% 25degC_NN_Pan18650PF.mat
data = input('file with the data: ','s');
container = load(data);

t = container.meas.Time;

% Feature 1: Voltage
Voltage = container.meas.Voltage;
N = size(Voltage,1);
% Feature scaling (mean normalization)
%Voltage_scaling = (Voltage - mean(Voltage))/max(Voltage);
Voltage_scaling = (Voltage - mean(Voltage))/std(Voltage);

% Feature 2: Current
Current = container.meas.Current;
% Feature scaling
Current_scaling = (Current - mean(Current))/abs(min(Current));
%Current_scaling = (Current - mean(Current))/std(Current);

% Feature 3: Temperature
Temperature = container.meas.Battery_Temp_degC;
% Feature scaling
%Temperature_scaling = (Temperature - mean(Temperature))/max(Temperature);
Temperature_scaling = (Temperature - mean(Temperature))/std(Temperature);

% Label: State of Charge
Ah_discharge = container.meas.Ah;
pAh = -1*Ah_discharge;
bias = (-1)*min(Ah_discharge); % 
SOC = 100*((Ah_discharge + bias)/bias);

% Saves all training data and labels (targets)
% The validation split is done by the Python code
% In this case, we will not with test data and test labels
% train_data = [Voltage_scaling Current_scaling Temperature_scaling];
% save('train_data.mat', 'train_data');
% save('train_label.mat', 'SOC');

% Test data and test labels 

% generate random integers
rng('shuffle')
r = randperm(N,round(0.2*N));% vector 9.612 indexes
r = r';
r = sort(r); % sorts elements in ascending order

% Test data and test labels
Voltage_test = Voltage_scaling(r); 
Current_test = Current_scaling(r);
Temperature_test = Temperature_scaling(r);
test_data = [Voltage_test Current_test Temperature_test];
save('test_data.mat', 'test_data');

SOC_label_test = SOC(r);
save('test_label.mat', 'SOC_label_test');

% Training data and labels
a = [1:N];
a(r)=[];
a = a';
%Voltage_train = Voltage(a);
Voltage_train = Voltage_scaling(a);
%Current_train = Current(a);
Current_train = Current_scaling(a);
%Temperature_train = Temperature(a);
Temperature_train = Temperature_scaling(a);
train_data = [Voltage_train Current_train Temperature_train];
save('train_data.mat', 'train_data');
save('train_data.csv', 'train_data')

SOC_label_train = SOC(a);
save('train_label.mat', 'SOC_label_train');
save('train_label.csv', 'SOC_label_train');

% A few plots to visualize the Drive Cycle
figure(1)
plot(SOC,Temperature,'r')  
xlabel('SOC (%)')
ylabel('Temperature (Celsius)')
grid
hold on
%pause

figure(2)
plot(t,Ah_discharge,'m')
xlabel('Time (seconds)')
ylabel('Amp-hours discharged')
hold on
grid
%pause

figure(3)
%plot(t,Voltage,'b')
plot(t,Voltage_scaling,'b')
xlabel('Time (seconds)')
ylabel('Voltage (V)')
grid
hold on
%pause

figure(4)
%plot(t,Current,'k')
plot(t,Current_scaling,'k')
xlabel('Time (seconds)')
ylabel('Current (A)')
grid
hold on
%pause

figure(5)
%plot(t,Temperature,'g')
plot(t,Temperature_scaling,'g')
xlabel('Time (seconds)')
ylabel('Temperature (Celsius)')
grid
hold on
%pause

figure(6)
plot(SOC,Voltage,'r')
xlabel('SOC (%)')
ylabel('Voltage (V)')
grid
hold on
%pause

%close all