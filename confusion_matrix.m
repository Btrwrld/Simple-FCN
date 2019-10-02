function matrix=confusion_matrix(Y_real, Y_pred)

   # Number of classes and data 
   num_classes = columns(Y_real);
   data_amount = rows(Y_real);

   # Initialize confusion matrix
   matrix = zeros(num_classes,num_classes);
   
   # Fill matrix
   for(i=[1:data_amount])
    [_, irow] = max(Y_real(i,:));
    [_, icol] = max(Y_pred(i,:));  
    matrix(irow, icol) += 1;
   endfor
   
   # Show the matrix
   disp("Confusion matrix:");
   disp(matrix);
   
   # Initialize precision, recall, and f1score
   precision = zeros(num_classes,1);
   recall = zeros(num_classes,1);
   f1score = zeros(num_classes,1);

   # Calculate values class-wise
   for(i=[1:num_classes])
    sumrow = sum(matrix(i,:));
    sumcol = sum(matrix(:,i));
    if(sumrow != 0 )
      recall(i) = matrix(i,i)/sumrow;
    endif
    if(sumcol != 0)
      precision(i) = matrix(i,i)/sumcol;
    endif
    if((precision(i) + recall(i)) != 0)
      f1score(i) = (2 * (precision(i) * recall(i) / (precision(i) + recall(i))));
    endif
   endfor
   
   # Show the resulting values
   disp("Precision");
   disp(precision);
   disp("Recall");
   disp(recall);
   disp("F1Score");
   disp(f1score);
   
endfunction

