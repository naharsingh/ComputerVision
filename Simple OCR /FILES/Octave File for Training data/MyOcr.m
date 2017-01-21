input_layer_size = 400;    
num_labels = 26;

data=load('d:\data.txt');
X=data(:,1:end-1);
Y=data(:,end);
m=size(X,1);

rand_indices = randperm(m);
sel=X(rand_indices(1:100),:);
displayData(sel);

fprintf('\nTraining One-vs-All Ligistic Regression....\n')

lambda=0.5;
[all_theta]=oneVsAll(X,Y,num_labels,lambda);

pred=predictOneVsAll(all_theta, X);
fprintf('\nTraining Set Accurary: %f\n', mean(double(pred==Y))*100);
fprintf('Exporting model.....\n');
save -ascii d:\learnedParams.txt all_theta;