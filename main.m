pkg load statistics
pkg load optim
close;

# Define parameters
ds_size = 1000;
test_val_size = 128;
num_clases = 4;
epochs = 120;
batch_size = 8;
lr = 0.01;
optim = "dg"; 
shape = "vertical";

train(shape, num_clases, batch_size, ds_size, lr, epochs, optim, false);

test(shape, ds_size, test_val_size, num_classes);





