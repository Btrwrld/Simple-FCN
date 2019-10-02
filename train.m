
function train(shape="vertical", num_classes=3, batch_size=8, ds_size=1000, lr = 0.005, epochs = 100, optim = "dg", keep_training = true)
    pkg load statistics
    pkg load optim
    close;

    model = FCN(2, num_classes, optim);
	if(keep_training)
        ws = glob(["./weights/" strcat(shape,  "*", num2str(ds_size), "*",num2str(num_classes)), "_*"]);
        if(size(ws)(1)==3)
            w1strfile = ws{1};
            w2strfile = ws{2};
            w3strfile = ws{3};
            W1 = csvread(w1strfile);
            W2 = csvread(w2strfile);
            W3 = csvread(w3strfile);

            model.layers{1}.weights = W1;
            model.layers{2}.weights = W2;
            model.layers{3}.weights = W3;
        endif
    endif


    # optim = "dg"; # cg(recommended lr 0.01) and dg available
    # Generate and plot training_samples
    [training_samples, training_targets] = create_data(ds_size, num_classes, shape);
    % plot_data(training_samples, training_targets);

    # Start training 

    [model, val_set] = model.train(training_samples, training_targets, epochs, batch_size, lr);

    # Write the weights in the associated files
    dlmwrite (strcat('./weights/',shape,'_ds_size:', num2str(ds_size),'_classes:',num2str(num_classes),'_W1.csv'), model.layers{1}.weights, ",");
    dlmwrite (strcat('./weights/',shape,'_ds_size:', num2str(ds_size),'_classes:',num2str(num_classes),'_W2.csv'), model.layers{2}.weights, ",");
    dlmwrite (strcat('./weights/',shape,'_ds_size:', num2str(ds_size),'_classes:',num2str(num_classes),'_W3.csv'), model.layers{3}.weights, ",");
endfunction