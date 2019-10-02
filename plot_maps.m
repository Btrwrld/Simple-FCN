function plot_maps(model,X,y,numclasses)


  ####################################################
  ##           Set Drawing Parameters               ##
  ####################################################

  colors={'k','r','b','m','g','c','y'};
  markers={'+','o','*','x','s','d','^','v','>','<'};

  ####################################################
  ##              Predict with model                ##
  ####################################################

  ## Create an image, where the color of the pixel is created by
  ## combining a bunch of colors representing each class, and the
  ## mixture is made with the probabilities.

  x=linspace(-1,1,256);
  [GX,GY]=meshgrid(x,x);
  FX = [GX(:) GY(:)];

  FZ = [];
  for(i=[1:rows(FX)])
            [Y_curr, fwd]=model.predict(FX(i, :));
            FZ = [FZ; Y_curr];

  endfor
  % FZ = model.predict(FX);

  # Normalize FZ
  FZ = FZ ./ sum(FZ,2);
  FZ = FZ';
 
  ## A figure with the winners
  [maxprob,maxk]=max(FZ);
   
  figure("name","Winner classes");
  
  winner=flip(uint8(reshape(maxk,size(GX))),1);
  cmap = [0,0,0; 1,0,0; 0,0,1; 0.5,0,0.5; 0,1,0; 0,0.85,0.85; 0.5,0.5,0.0];
  wimg=ind2rgb(winner,[0,0,0;cmap]);
  imshow(wimg);
   
  ## A figure with the weighted winners
  figure();
   
  ccmap = cmap(1:numclasses,:);
  cwimg = ccmap'*FZ;
  redChnl   = reshape(cwimg(1,:),size(GX));
  greenChnl = reshape(cwimg(2,:),size(GX));
  blueChnl  = reshape(cwimg(3,:),size(GX));
   
  mixed = flip(cat(3,redChnl,greenChnl,blueChnl),1);
  imshow(mixed);
  title("Classes by probability.");
   
   
  ## Show the output probability for each class separately
  for (kk=[1:numclasses])
    figure(); hold off;
    x=-1:0.05:1;
    [GX,GY]=meshgrid(x,x);
    FX = [GX(:) GY(:)];

    FZ = [];
    for(i=[1:rows(FX)])
        [Y_curr, fwd]=model.predict(FX(i, :));
        FZ = [FZ; Y_curr];

    endfor
    FZ = FZ';
    GZ = reshape(FZ(kk,:),size(GX));
    h = surface(GX,GY,GZ);
    daspect([1,1,1]);
    hold on;

    # Offset surface z data so the scatter can be on top.
    z = get(h,'ZData');
    set(h,'ZData',z-10);
    
    P = X(y(:,kk)==1,:);
    #scatter(X(y(:,kk)==1,1),X(y(:,kk)==1,2),[],colors{kk},markers{kk});
    scatter3(P(:,1), P(:,2),y(y(:,kk)==1)/kk,
           [],colors{kk},'d',"filled");
    #scatter3(X(y(:,kk)!=1,1),X(y(:,kk)!=1,2),y(y(:,kk)!=1)*0,
             #[],"k","+","filled");
   
    xlabel("x");
    ylabel("y");
    title(cstrcat("Class ",num2str(kk)));
  endfor;
endfunction;