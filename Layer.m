classdef Layer

    # Define the function and its derivative as tributes
    properties

        # Define important properties
        neurons = 0;
        weights = [];
        funct = 0;
        optim = "";
        epsilon = 0.001;
        limit = 1;

    endproperties


    methods

        # Constructor
        function layer = Layer(input_size, output_size, funct, optim)

            # Store the number of neurons per layer
            layer.neurons = input_size + 1;
            # The +1 is to take the bias into account
            layer.weights = rand([layer.neurons output_size]);
            # Save the activation function and its derivative
            layer.funct = funct;
            layer.optim = optim;

        endfunction

        # Takes an row vector as input adds the bias and forwards 
        # it into the layer
        function y = forward(layer, x)
        
            # Add the bias
            x = [x 1];
            # Forward
            y = layer.funct.f(x * layer.weights);

        endfunction

        # Takes the inputs of the network 
        function [x_grad w_grad] = backward(layer, grad, inputs, outputs)

            # Calc the gradient using a defined optim
            switch(layer.optim)

                case "cg"
                    g = cg_min(layer.funct.fc, layer.funct.dfc, outputs', [1 layer.epsilon 0 layer.limit])';
                    g = g ./ -norm(g); # Make g a unitary vector since all we need is the direction

                case "dg"
                    g = layer.funct.df(outputs);

            endswitch

            # Calc the gradient of the actv function
            # IxO =IxO  .*     IxO
            grad = grad .* g;
            
            # Calc weight gradient and input grad
            #  NxO =  NxI    * IxO                     
            w_grad = inputs' * grad;
            # I+1xI=   I+1xO       *  OxI
            x_grad = layer.weights * grad';

        endfunction

    endmethods

endclassdef