
pkg load statistics
pkg load optim
close;

# Define parameters
ds_size = 1000;
test_val_size = 100;
num_clases = 3;
epochs = 100;
batch_size = 8;
lr = 0.0005;
optim = "cg"; # cg(recommended lr 0.01) and dg available
shape = "vertical";

# Generate and plot training_samples
[training_samples, training_targets] = create_data(ds_size, num_clases, shape);
plot_data(training_samples, training_targets);

# Start training 
model = FCN(2, num_clases, optim);
[model, val_set] = model.train(training_samples, training_targets, epochs, batch_size, lr);

input("Press enter to continue with tests\n")
[test_samples, test_targets] = create_data(test_val_size, num_clases, shape);
predictions = [];
# Make predictions
for i = 1:test_val_size

    [p _] = model.predict(test_samples(i, :));
    p 
    test_targets(i,:)
    disp("\n\n")

endfor


















