clc;
clear
close all;
rng default;

X = zeros(6000,900);
Y = zeros(6000,1);
% cursor = 1;
% for i = 1:15
%     for j = 1:400
%         vector = 3*randn(1,900);
%         vector(i*40:i*40+40-1) = 10+2*randn(1,40);
%         X(cursor,:) = vector;
%         Y(cursor) = i;
%         cursor = cursor + 1;
%     end
% end
cursor = 1;
labelCounter = 1;
for i = 1:15
    stopFlg = 0;
    for j = 1:400
        vector = 3*randn(1,900);
        vector(i*40:i*40+40-1) = 10+2*randn(1,40);
        X(cursor,:) = vector;
        if i == 3 
            Y(cursor) = 1;
            stopFlg = 1;
        else
            Y(cursor) = labelCounter;
        end
%         Y(cursor) = labelCounter;
        cursor = cursor + 1;
    end
    if stopFlg == 1
        stopFlg = 0;
    else
        labelCounter = labelCounter + 1;        
    end
end
unknownX = X(10*400+1:end,:);
unknownY = Y(10*400+1:end,:);
X = X(1:10*400,:);
Y = Y(1:10*400,:);
tUnknownX = reshape(unknownX',[30,30,1,size(unknownX,1)]);

randSeries = randperm(size(X,1));
X = X(randSeries,:);
Y = Y(randSeries,:);
cX = X(floor(0.6*size(X,1)):end,:);
cY = Y(floor(0.6*size(Y,1)):end,:);
X = X(1:floor(0.6*size(X,1))-1,:);
Y = Y(1:floor(0.6*size(Y,1))-1,:);
tX = reshape(X',[30,30,1,size(X,1)]);
tcX = reshape(cX',[30,30,1,size(cX,1)]);

inputSize = [size(tX,1) size(tX,2) size(tX,3)];
numClasses = size(unique(Y(:,1)),1);

layers = [
    imageInputLayer(inputSize, 'Name', 'input','Mean', mean(tX,4))    
    convolution2dLayer(2,2, 'Name', 'conv2d_1')
    batchNormalizationLayer('Name', 'batchNorm_1')
    reluLayer('Name', 'relu_1')
    tensorVectorLayer('Flatten')   
    FCLayer(1682,numClasses,'Fingerprints',[])
    %softmaxLayer('Name', 'softmax_1')
    yxSoftmax('softmax_1')
    classificationLayer('Name', 'classify_1')
    ];
lgraph = layerGraph(layers);

XTrain = tX;
YTrain = Y;
numEpochs = 4;
miniBatchSize = 20;
plots = "training-progress";
executionEnvironment = "auto";
totalIters = 0;

if plots == "training-progress"
    figure(10);
    lineLossTrain = animatedline('Color','#0072BD','lineWidth',1.5);
    lineClassificationLoss = animatedline('Color','#EDB120','lineWidth',1.5);
      
    ylim([-inf inf])
    xlabel("Iteration")
    ylabel("Loss")
    legend('Loss','classificationLoss');
    grid on;
    
    figure(11);  
    lineCVAccuracy = animatedline('Color','#D95319','lineWidth',1.5);
    lineMultiKernelLoss = animatedline('Color','#7E2F8E','lineWidth',1.5);
    ylim([0 1])
    xlabel("Iteration")
    ylabel("Acc.")    
    legend('CV Acc.','Avg. Kernel dist.');
    grid on;    
end
L2RegularizationFactor = 0.01;
initialLearnRate = 0.01;
decay = 0.01;
momentum = 0.9;
velocities = [];
learnRates = [];
momentums = [];
gradientMasks = [];
numObservations = numel(YTrain);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
iteration = 0;
start = tic;
classes = categorical(YTrain);
% lgraph2 = layerGraph(net); % Also collect old weights
% % OR:
lgraph2 = lgraph; % No old weights
lgraph2 = lgraph2.removeLayers('classify_1');
dlnet = dlnetwork(lgraph2);
% Loop over epochs.
for epoch = 1:numEpochs
    idx = randperm(numel(YTrain));
    XTrain = XTrain(:,:,:,idx);
    YTrain = YTrain(idx); 
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        totalIters = totalIters + 1;
        % Read mini-batch of data and convert the labels to dummy
        % variables.
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        Xb = XTrain(:,:,:,idx);
        Yb = zeros(numClasses, miniBatchSize, 'single');
        for c = 1:numClasses
            Yb(c,YTrain(idx)==(c)) = 1;
        end
        % Convert mini-batch of data to dlarray.
        dlX = dlarray(single(Xb),'SSCB');
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,state,loss] = dlfeval(@modelGradientsOnWeights,dlnet,dlX,Yb);
        dlnet.State = state;
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate;%/(1 + decay*iteration);
        % Update the network parameters using the SGDM optimizer.
        %[dlnet, velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);
        % Update the network parameters using the SGD optimizer.
        %dlnet = dlupdate(@sgdFunction,dlnet,gradients);
        if isempty(velocities)
            velocities = packScalar(gradients, 0);
            learnRates = packScalar(gradients, learnRate);
%             momentums = packScalar(gradients, momentum); % Only for SGD optimizer!
            momentums = packScalar(gradients, 0);
            L2Foctors = packScalar(gradients, 0);            
            gradientMasks = packScalar(gradients, 1);   
%             % Let's lock some weights
%             for k = 1:2
%                 gradientMasks.Value{k}=dlarray(zeros(size(gradientMasks.Value{k})));
%             end
        end

%         [dlnet, velocities] = dlupdate(@sgdmFunctionL2, ...
%             dlnet, gradients, velocities, ...
%             learnRates, momentums, L2Foctors, gradientMasks);
        totalIterPk = packScalar(gradients, totalIters);
        [dlnet, velocities, momentums] = dlupdate(@adamFunction, ...
                    dlnet, gradients, velocities, ...
                    learnRates, momentums, L2Foctors, gradientMasks, ...
                    totalIterPk);    

        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            XTest = tcX;
            YTest = categorical(cY);
            if mod(iteration,5) == 0 
                accuracy = cvAccuracy(dlnet, XTest,YTest,miniBatchSize,executionEnvironment);
                addpoints(lineCVAccuracy,iteration, accuracy);
            end
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end

function accuracy = cvAccuracy(dlnet, XTest, YTest, miniBatchSize, executionEnvironment)
    dlXTest = dlarray(XTest,'SSCB');
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        dlXTest = gpuArray(dlXTest);
    end
    dlYPred = modelPredictions(dlnet,dlXTest,miniBatchSize);
    [~,idx] = max(extractdata(dlYPred),[],1);
    YPred = categorical(idx);
    accuracy = mean(YPred(:) == YTest(:));
end

function dlYPred = modelPredictions(dlnet,dlX,miniBatchSize)
    numObservations = size(dlX,4);
    numIterations = ceil(numObservations / miniBatchSize);
    numClasses = size(dlnet.Layers(end-1).Weights,1);
    dlYPred = zeros(numClasses,numObservations,'like',dlX);
    for i = 1:numIterations
        idx = (i-1)*miniBatchSize+1:min(i*miniBatchSize,numObservations);
        dlYPred(:,idx) = predict(dlnet,dlX(:,:,:,idx));
    end
end

function [gradients,state,loss] = modelGradientsOnWeights(dlnet,dlX,Y)
%   %This is only used with softmax of matlab which only applies softmax
%   on 'C' and 'B' channels.
    [rawPredictions,state] = forward(dlnet,dlX,'Outputs', 'Fingerprints');
    dlYPred = softmax(dlarray(squeeze(rawPredictions),'CB'));
%     [dlYPred,state] = forward(dlnet,dlX);
    penalty = 0;
    scalarL2Factor = 0;
    if scalarL2Factor ~= 0
        paramLst = dlnet.Learnables.Value;
        for i = 1:size(paramLst,1)
            penalty = penalty + sum((paramLst{i}(:)).^2);
        end
    end
    loss = crossentropy(squeeze(dlYPred),Y) + scalarL2Factor*penalty;
    gradients = dlgradient(loss,dlnet.Learnables);
    %gradients = dlgradient(loss,dlnet.Learnables(4,:));
end

function [params,velocityUpdates,momentumUpdate] = adamFunction(params, rawParamGradients,...
    velocities, learnRates, momentums, L2Foctors, gradientMasks, iters)
    % https://arxiv.org/pdf/2010.07468.pdf %%AdaBelief
    % https://arxiv.org/pdf/1711.05101.pdf  %%DeCoupled Weight Decay 
    b1 = 0.9; 
    b2 = 0.999;
    e = 1e-16;
    curIter = iters(:);
    curIter = curIter(1);
    
    gt = rawParamGradients;
    mt = (momentums.*b1 + ((1-b1)).*gt);
    vt = (velocities.*b2 + ((1-b2)).*((gt-mt).^2));

     momentumUpdate = mt;
     velocityUpdates = vt;
    h_mt = mt./(1-b1.^curIter);
    h_vt = (vt+e)./(1-b2.^curIter);
    params = params - 0.001.*(mt./(sqrt(vt)+e)).*gradientMasks...
        -L2Foctors.*params.*gradientMasks;
%     params = params - 0.001.*(h_mt./(sqrt(h_vt)+e)).*gradientMasks...
%         -L2Foctors.*params.*gradientMasks;

end

function param = sgdFunction(param,paramGradient)
    learnRate = 0.01;
    param = param - learnRate.*paramGradient;
end

function [params, velocityUpdates] = sgdmFunction(params, paramGradients,...
    velocities, learnRates, momentums)
% https://towardsdatascience.com/stochastic-gradient-descent-momentum-explanation-8548a1cd264e
    velocityUpdates = momentums.*velocities+learnRates.*paramGradients;
    params = params - velocityUpdates;
end

function [params, velocityUpdates] = sgdmFunctionL2(params, rawParamGradients,...
    velocities, learnRates, momentums, L2Foctors, gradientMasks)
% https://towardsdatascience.com/stochastic-gradient-descent-momentum-explanation-8548a1cd264e
% https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261
    paramGradients = rawParamGradients + 2*L2Foctors.*params;
    velocityUpdates = momentums.*velocities+learnRates.*paramGradients;
    params = params - (velocityUpdates).*gradientMasks;
end

function tabVars = packScalar(target, scalar)
% The matlabs' silly design results in such a strange function
    tabVars = target;
    for row = 1:size(tabVars(:,3),1)
        tabVars{row,3} = {...
            dlarray(...
            ones(size(tabVars.Value{row})).*scalar...%ones(size(tabVars(row,3).Value{1,1})).*scalar...
            )...
            };
    end
end
