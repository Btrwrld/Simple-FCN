classdef FCN

    # Define the function and its derivative as tributes
    properties 

        # Store the activation fuctions
        func;
        # Store the layers
        layers = {};
        # Store the target function and its derivative
        target; 
        # Set the current mode
        isTraining = false;

    endproperties


    methods

        # Constructor
        function fcn = FCN(input_size, output_size, optim)
            # Create a function object
            fcn.func = Function();

            # Add layers
            fcn.layers{1} = Layer(input_size , 10          , fcn.func.Sigmoid, optim);
            fcn.layers{2} = Layer( 10         , output_size , fcn.func.Softmax, optim);

            # Set the target function as least squares
            fcn.target = fcn.func.LeastSquares;

            disp("FCN created!")
        endfunction

        function [y, forward_values] = predict(model, X)

            # usage predict(X)
            #
            # This function propagates the input X through the neural network to
            # predict the output vector y, given the layer in the nn
            #
            # X: Input row vector

            # Define the forward values container and concatenate the bias
            forward_values = {};

            partial_forward_values{1} = [X 1];

            # Calc and store the forward values of the net
            for i = 1:length(model.layers)
        
                X = model.layers{i}.forward(X);
                partial_forward_values{i + 1} = [X 1];
            endfor
        

            # If we are on training mode, store the result for 
            # the last forward
            if (model.isTraining)
                forward_values = partial_forward_values;
            endif

            # y is a row vector
            y = X;

        endfunction

        function [fcn validation_set] = train(model, X, Y, epochs, batch_size, lr)

            # usage train(X,Y, epochs, batch_size, lr)
            #
            # This function takes the training set X where each column is a sample
            # and it's respective targets
            #
            # X: training set holding on the rows the input data
            # Y: labels of the training set as a column vector
            # epochs: Number of epochs needed to train
            # batch_size: size of training batch
            # lr: learning rate


            # Define a validation set as a ~30% of the input data
            val_rows = randperm( size(X)(1), floor(size(X)(1) * 0.3));
            validation_set = X( val_rows, : );
            validation_target = Y( val_rows, : );
            # Remove these rows from training data
            X( val_rows, : ) = [];
            Y( val_rows, : ) = [];

            figure(1);
            plot_data(X, Y);
            title("Training Data Set");


            # Set the model to training mode to store forward prop values
            model.isTraining = true;

            # Plot error 
            figure(2);
            ylabel('Batch error')
            xlabel('Epoch')
            title('Error evolution')
            hold on;
            err = [];
            ep = [];

            #  Repeat training epochs times
            for i = 1:epochs

                # define the epoch training set and target
                training_set = X;
                targets = Y;
                predictions = [];

                # Go throung all training samples
                while (length(training_set) > 0)
                    
                    # Calc the gradient 
                    [model training_set targets] = model.gradtarget(training_set, targets, batch_size, lr);

                endwhile

                # Validate 
                for j = 1:length(val_rows)

                    # Make the prediction for the validation set sample
                    [pred, _] = model.predict( validation_set(j, : ) );
                    # Store it
                    predictions = [predictions; pred];
                    
                endfor        

                # Plot the error
                ep = [ep i];
                e = model.target.f(predictions, validation_target)./length(val_rows);
                err = [err; e sum(e)];

                fprintf("Epoch %i\t Mean error %f\n", i, sum(e));
                figure(2, "name", strcat("Loss evolution for learning rate: ", num2str(lr)));
                #Plot error
                colors  = ['k','r','b','m','g','c','y'];
                c = 0;
                hold on;
                l = {};
                for i = 1:(size(err)(2) - 1)

                    plot(ep, err(:, i), colors(i), "linewidth", 3);
                    l{i} = strcat("Error for class  " , int2str(i));
                    c = i;
                endfor
                c += 1;
                l{c} = "Mean error";
                plot(ep, err(:, end), colors(c),"linewidth", 3);
                legend (l, "location", "northeastoutside");
                refresh();



            endfor
            model.isTraining = false;

            #Plot error
            colors  = ['k','r','b','m','g','c','y'];
            c = 0;
            hold on;
            l = {};
            for i = 1:(size(err)(2) - 1)

                plot(ep, err(:, i), colors(i), "linewidth", 3);
                l{i} = strcat("Error for class  " , int2str(i));
                c = i;
            endfor
            c += 1;
            l{c} = "Mean error";
            plot(ep, err(:, end), colors(c),"linewidth", 3);
            legend (l, "location", "northeastoutside");

            # Store and return changes in model
            fcn = model;

        endfunction

        function [fcn training_set targets] = gradtarget(model, X, Y, batch_size = 1e12, lr)

            # usage gradtarget(model, X, Y, batchSize)
            #
            # This function evaluates the gradient of the target function on
            # the model layers
            #
            # X :training set holding on the rows the input data, plus a final row
            # equal to 1
            # Y :labels of the training set
            # batchSize: Size of the batch used to compute the gradient
            # lr: learning rate


            # If the batch size is grater than the amount of samples(num rows) then
            # use all samples
            batch_size = min(size(X)(1), batch_size);

            # Get the column number of the randomly selected samples by
            # generating a random permutation from 1:total_samples and then
            # select the first batch_size samples
            batch = randperm( size(X)(1), batch_size );

            # Define container variables
            forward_values = {};
            predictions = [];
            target = [];
            
            # Predict the values for the batch
            for i = 1:batch_size

                # Predict and store
                [pred, forward_values{i}] = model.predict( X( batch(i), : ) );
                # Fill predictions and target matrix, aech row is a sample
                predictions = [predictions; pred];
                target = [target; Y( batch(i), : )];

            endfor


            # Calc the grad
            # Output grad = target grad * input = 1xOutputDim
            grad = predictions - target;
            w_grad = 0;
            x_grad = [];

            # Calc gradient for the network
            for i = length(model.layers):-1:1
                w_grad = 0;
                x_grad = [];
                
                # Calc partial grad using each sample in the batch
                for j = 1:length(forward_values)
                    # Calc the gradient of x and w for aech sample in the lsayer i
                    [x w] = model.layers{i}.backward(   grad(j, :), 
                                                        forward_values{j}{i}, 
                                                        forward_values{j}{i + 1}(:, 1:end-1));# Remove bias

                    # Accumulate gradient
                    w_grad += w; 
                    x_grad(j,:) = x'; 
                    
                endfor

                # Remove the bias from x_grad if the current layer is the output layer
                x_grad = x_grad(:, 1:end-1);


                # update weigths remove bias
                model.layers{i}.weights -= lr .* w_grad;

                # Update grad 
                grad = x_grad;
        
            endfor

            # Remove used samples from the training set and targets
            training_set = X;
            targets = Y;
            training_set(batch, :) = [];
            targets(batch, :) = [];

            # Store and return changes in model
            fcn = model;

        endfunction

    endmethods

endclassdef