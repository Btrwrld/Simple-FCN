pkg load statistics
pkg load optim
close;

# Define parameters
ds_size = 1000;
test_val_size = 100;
num_clases = 6;
epochs = 120;
batch_size = 8;
lr = 0.01;
optim = "dg"; # cg(recommended lr 0.01) and dg available
shape = "radial";

train(shape, num_clases, batch_size, ds_size, lr, epochs, optim, false);

% # Generate and plot training_samples
% [training_samples, training_targets] = create_data(ds_size, num_clases, shape);
% plot_data(training_samples, training_targets);

% # Start training 
% model = FCN(2, num_clases, optim);
% [model, val_set] = model.train(training_samples, training_targets, epochs, batch_size, lr);

% # Write the weights in the associated files
% dlmwrite (strcat('./weights/',shape,'_ds_size:', num2str(ds_size),'_classes:',num2str(num_clases),'_W1.csv'), model.layers{1}.weights, ",");
% dlmwrite (strcat('./weights/',shape,'_ds_size:', num2str(ds_size),'_classes:',num2str(num_clases),'_W2.csv'), model.layers{2}.weights, ",");
% dlmwrite (strcat('./weights/',shape,'_ds_size:', num2str(ds_size),'_classes:',num2str(num_clases),'_W3.csv'), model.layers{3}.weights, ",");

% input("Press enter to continue with tests\n")
% [test_samples, test_targets] = create_data(test_val_size, num_clases, shape);
% predictions = [];
% # Make predictions
% for i = 1:test_val_size

%     [p _] = model.predict(test_samples(i, :));
%     p 
%     test_targets(i,:)
%     disp("\n\n")

% endfor





