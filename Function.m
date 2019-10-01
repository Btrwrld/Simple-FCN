classdef Function

    # Define the function and its derivative as tributes
    properties
        # Functions
        Sigmoid;
        Softmax;
        LeastSquares;
        ReLu;

    endproperties


    methods

        # Constructor
        function F = Function()

            # Define the available functions

            # Sigmoid
            F.Sigmoid.f = @(x) 1 ./ (1 + exp(-x));
            F.Sigmoid.df = @(x) F.Sigmoid.f(x) .* ( 1 - F.Sigmoid.f(x) );
            # Cells
            F.Sigmoid.fc = @(x) F.Sigmoid.f(x{1});
            F.Sigmoid.dfc = @(x) F.Sigmoid.df(x{1}); 


            # ReLu
            F.ReLu.f = @(x) (x > 0) .* x;
            F.ReLu.df = @(x) (x > 0);
            # Cells
            F.ReLu.fc = @(x) F.ReLu.f(x{1});
            F.ReLu.dfc = @(x) F.ReLu.df(x{1});


            # Softmax
            # This one is numerically stable since its normalized by the maximum 
            # and converges to 0 instead of Nan
            F.Softmax.f =  @(x) exp(x - max(x)) ./ sum(exp(x - max(x)));    
            F.Softmax.df =  @(x) F.Softmax.f(x) .* ( 1 - F.Softmax.f(x) );
            # Cells
            F.Softmax.fc =  @(x) F.Softmax.f(x{1});
            F.Softmax.dfc =  @(x) F.Softmax.df(x{1});


            # Least squares
            F.LeastSquares.f = @(y_hat, y) 1/2 * sum( (y_hat - y) .* (y_hat - y) );
            F.LeastSquares.df = @(y_hat, y) sum( (y_hat - y) );
            # Cells
            F.LeastSquares.fc = @(y_hat_n_y) F.LeastSquares.f(y_hat_n_y{1}, y_hat_n_y{2});
            F.LeastSquares.dfc = @(y_hat_n_y) F.LeastSquares.df(y_hat_n_y{1}, y_hat_n_y{2});
            

        endfunction
        

    endmethods

endclassdef