#!/usr/bin/octave-cli

# usage train(shape, num_classes, batch_size, ds_size, lr, epochs, optim, keep_training)
# 
# This functions trains the neural network and stores the result in the weights folder
# 
# shape: shape to train with. Default = "vertical"
# num_classes: number of classes to generate. Default = 3
# batchSize: size of batch for the  gradient descent. Default = 8
# ds_size: size of the data set to generate. Default = 1000
# lr: learing rate for the weight calculation. Defaul = 0.01
# epochs: maximum amount of epochs allowed. Default = 100
# optim: optimization method. "dg" = gradient descent, "cg" = conjugate gradients. Default = "dg"
# keep_training: whether or not the training should resume from previously calculated weights. Default = true

function train(shape="vertical", num_classes=3, batch_size=8, ds_size=1000, lr = 0.01, epochs = 100, optim = "dg", keep_training = true)
    pkg load statistics
    pkg load optim
    close;

    model = FCN(2, num_classes, optim);
	if(keep_training)
        ws = glob(["./weights/" strcat(shape,  "*", num2str(ds_size), "*",num2str(num_classes)), "_*"]);
        if(size(ws)(1)==3)
            w1strfile = ws{1};
            w2strfile = ws{2};

            W1 = csvread(w1strfile);
            W2 = csvread(w2strfile);


            model.layers{1}.weights = W1;
            model.layers{2}.weights = W2;

        endif
    endif


    # Generate and plot training_samples
    [training_samples, training_targets] = create_data(ds_size, num_classes, shape);

    # Start training 

    [model, val_set] = model.train(training_samples, training_targets, epochs, batch_size, lr);

    # Write the weights to the associated files
    dlmwrite (strcat('./weights/',shape,'_ds_size:', num2str(ds_size),'_classes:',num2str(num_classes),'_W1.csv'), model.layers{1}.weights, ",");
    dlmwrite (strcat('./weights/',shape,'_ds_size:', num2str(ds_size),'_classes:',num2str(num_classes),'_W2.csv'), model.layers{2}.weights, ",");

endfunction