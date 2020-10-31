%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Please be noted that part of this code and some critical hyperparams are contributed by juntang-zhuang
w.r.t.: https://github.com/juntang-zhuang/Adabelief-Optimizer/issues/22
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;
close all;
clear;
rng default;
%%----------- Check Point 0:  
% Please download the dataset from
% https://drive.google.com/uc?export=download&id=1N-eImoAA3QFPu3cBJd-1WIUH0cqw2RoT
% Or
% https://ieee-dataport.org/open-access/24-hour-signal-recording-dataset-labels-cybersecurity-and-iot
load('adsb_records_qt.mat');
%%%%---------- End of Check Point 0
payloadMatrix = reshape(payloadMatrix', ...
    length(payloadMatrix)/length(msgIdLst), length(msgIdLst))';
rawIMatrix = reshape(rawIMatrix', ...
    length(rawIMatrix)/length(msgIdLst), length(msgIdLst))';
rawQMatrix = reshape(rawQMatrix', ...
    length(rawQMatrix)/length(msgIdLst), length(msgIdLst))';
rawCompMatrix = rawIMatrix + rawQMatrix.*1j;
if size(rawCompMatrix,2) < 1024
    appendingBits = (ceil(sqrt(size(rawCompMatrix,2))))^2 - size(rawCompMatrix,2);
    rawCompMatrix = [rawCompMatrix, zeros(size(rawCompMatrix,1), appendingBits)];
else
   rawCompMatrix = rawCompMatrix(:,1:1024); 
end
dateTimeLst = datetime(uint64(timeStampLst),'ConvertFrom','posixtime','TimeZone','America/New_York','TicksPerSecond',1e3,'Format','dd-MMM-yyyy HH:mm:ss.SSS');
uIcao = unique(icaoLst);
c = countmember(uIcao,icaoLst);
icaoOccurTb = [uIcao,c];
icaoOccurTb = sortrows(icaoOccurTb,2,'descend');
cond1 = icaoOccurTb(:,2)>=300;
cond2 = icaoOccurTb(:,2)<=5000;
cond = logical(cond1.*cond2);
selectedPlanes = icaoOccurTb(cond,:);
unknowPlanes = icaoOccurTb(~cond,:);
allTrainData = [icaoLst, abs(rawCompMatrix)];
selectedBasebandData = [];
selectedRawCompData = [];
unknownBasebandData = [];
unknownRawCompData = [];
minTrainChance = 100;
maxTrainChance = 500;
for i = 1:size(selectedPlanes,1)
    selection = allTrainData(:,1)==selectedPlanes(i,1);
    localBaseband = allTrainData(selection,:);
    localComplex = rawCompMatrix(selection,:);
    localAngles = (angle(localComplex(:,:))); 
    
    if size(localBaseband,1) < minTrainChance
        continue;
    elseif size(localBaseband,1) >= maxTrainChance
        rndSeq = randperm(size(localBaseband,1));
        rndSeq = rndSeq(1:maxTrainChance);
        localBaseband = localBaseband(rndSeq,:);
        localComplex = localComplex(rndSeq,:);
    else
        %Nothing to do
    end
    selectedBasebandData = [selectedBasebandData; localBaseband];
    selectedRawCompData = [selectedRawCompData; localComplex];    
end
for i = 1:size(unknowPlanes,1)
    selection = allTrainData(:,1)==unknowPlanes(i,1);
    localBaseband = allTrainData(selection,:);
    localComplex = rawCompMatrix(selection,:);
    localAngles = (angle(localComplex(:,:)));
    unknownBasebandData = [unknownBasebandData; localBaseband];
    unknownRawCompData = [unknownRawCompData; localComplex];    
end
randSeries = randperm(size(selectedBasebandData,1));
selectedBasebandData = selectedBasebandData(randSeries,:);
selectedRawCompData = selectedRawCompData(randSeries,:);
randSeries = randperm(size(unknownBasebandData,1));
unknownBasebandData = unknownBasebandData(randSeries,:);
unknownRawCompData = unknownRawCompData(randSeries,:);
% 
% for i = 1:size(selectedBasebandData,1)
%     mu = mean(selectedBasebandData(i,2:end));
%     sigma = std(selectedBasebandData(i,2:end));
%     
%     % Use probabilistic filtering to restore the rational signal
%     mask = selectedBasebandData(i,2:end) <= mu + 3*sigma;
%     cleanInput = selectedBasebandData(i,2:end).*mask;
%     selectedBasebandData(i,2:end) = cleanInput;
% end
% for i = 1:size(selectedRawCompData,1)
%     muReal = mean(real(selectedRawCompData(i,1:end)));
%     sigmaReal = std(real(selectedRawCompData(i,1:end)));
%     muImag = mean(imag(selectedRawCompData(i,1:end)));
%     sigmaImag = std(imag(selectedRawCompData(i,1:end)));
%     
%     % Use probabilistic filtering to restore the rational signal
%     maskReal = abs(real(selectedRawCompData(i,1:end))) <= muReal + 3*sigmaReal;
%     cleanInputReal = real(selectedRawCompData(i,1:end)).*maskReal;
%     
%     maskImag = abs(imag(selectedRawCompData(i,1:end))) <= muImag + 3*sigmaImag;
%     cleanInputImag = imag(selectedRawCompData(i,1:end)).*maskImag;
%     
%     selectedRawCompData(i,1:end) = cleanInputReal+cleanInputImag.*1j;    
% end
[X,cX,Y,cY] = makeDataTensor(selectedBasebandData,selectedRawCompData);
[uX,cuX,uY,cuY] = makeDataTensor(unknownBasebandData,unknownRawCompData);
lookUpTab = [unique([Y;cY]),[1:length(unique([Y;cY]))]'];
Y2 = Y;
for i = 1:size(Y)
    Y2(i) = lookUpTab(find(lookUpTab(:,1) == Y(i)),2);
end
cY2 = cY;
for i = 1:size(cY)
    cY2(i) = lookUpTab(find(lookUpTab(:,1) == cY(i)),2);
end
Y = Y2;
cY = cY2;
inputSize = [size(X,1) size(X,2) size(X,3)];
numClasses = size(unique(selectedBasebandData(:,1)),1);
layers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Mean', 0)
    convolution2dLayer(5,10, 'Name', 'conv2d_1')
    batchNormalizationLayer('Name', 'batchNorm_1')
    reluLayer('Name', 'relu_1')
    convolution2dLayer(3, 10, 'Padding', 1, 'Name', 'conv2d_2')
    reluLayer('Name', 'relu_2')    
    convolution2dLayer(3, 10, 'Padding', 1, 'Name', 'conv2d_3')
    reluLayer('Name', 'relu_3')    
    depthConcatenationLayer(2,'Name','add_1')    
    fullyConnectedLayer(numClasses, 'Name', 'fc_bf_fp') % 11th
    
%%----------- Check Point 1:  
%% Here you can specify whether to use regular dense layer 
%% or zero-bias dense layer
%      fullyConnectedLayer(numClasses, 'Name', 'Fingerprints') 
    zeroBiasFCLayer(numClasses,numClasses,'Fingerprints',[])
%%----------- End of Check Point 1:         
    yxSoftmax('softmax_1')
    classificationLayer('Name', 'classify_1')
    ];
lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'relu_1', 'add_1/in2');
plot(lgraph);
XTrain = X;
YTrain = Y;
numEpochs = 3;
miniBatchSize = 128;
plots = "training-progress";
executionEnvironment = "auto";
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
    ylim([0 1.1])
    xlabel("Iteration")
    ylabel("Loss")    
    legend('CV Acc.','Avg. Kernel dist.');
    grid on;    
end
L2RegularizationFactor = 0.0;
initialLearnRate = 0.001;
decay = 0.01;
momentumSGD = 0.9;
velocities = [];
learnRates = [];
momentums = [];
gradientMasks = [];
numObservations = numel(YTrain);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
iteration = 0;
start = tic;
classes = categorical(YTrain);
lgraph2 = lgraph; % No old weights
lgraph2 = lgraph2.removeLayers('classify_1');
dlnet = dlnetwork(lgraph2);
% Loop over epochs.
totalIters = 0;
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
        [gradients,state,loss,classificationLoss] = dlfeval(@modelGradientsOnWeights,dlnet,dlX,Yb);
%         [gradients,state,loss] = dlfeval(@modelGradientsOnWeights,dlnet,dlX,Yb);        
        dlnet.State = state;
        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        % Update the network parameters using the SGDM optimizer.
        %[dlnet, velocity] = sgdmupdate(dlnet, gradients, velocity, learnRate, momentum);
        % Update the network parameters using the SGD optimizer.
        %dlnet = dlupdate(@sgdFunction,dlnet,gradients);
        if isempty(velocities)
            velocities = packScalar(gradients, 0);
            learnRates = packScalar(gradients, learnRate);
%             momentumSGDs = packScalar(gradients, momentumSGD);
            momentums = packScalar(gradients, 0);
            L2Foctors = packScalar(gradients, L2RegularizationFactor);            
            wd = packScalar(gradients, L2RegularizationFactor);  
            gradientMasks = packScalar(gradients, 1);   
%             % Let's lock some weights
%             for k = 1:2
%                 gradientMasks.Value{k}=dlarray(zeros(size(gradientMasks.Value{k})));
%             end
        end
%%%%----------- Check Point 2:  
%%%% Here you can specify which optimizer to use, 
%         [dlnet, velocities] = dlupdate(@sgdmFunctionL2, ...
%             dlnet, gradients, velocities, ...
%             learnRates, momentumSGDs, L2Foctors, gradientMasks); % This is
%             % the famous SGD with momentum
        totalIterInPackage = packScalar(gradients, totalIters); % We have to make this...
                                                         % stupid data
                                                         % structure but it
                                                         % only contains
                                                         % the number of
                                                         % iterations
        [dlnet, velocities, momentums] = dlupdate(@adamFunction, ...
                    dlnet, gradients, velocities, ...
                    learnRates, momentums, wd, gradientMasks, ...
                    totalIterInPackage);        
%         [dlnet] = dlupdate(@sgdFunction, ...
%             dlnet, gradients); % the vanilla
%%%%-----------End of Check Point 2 
        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            XTest = cX;
            YTest = categorical(cY);
            if mod(iteration,5) == 0 
                accuracy = cvAccuracy(dlnet, XTest,YTest,miniBatchSize,executionEnvironment,0);
                addpoints(lineCVAccuracy,iteration, accuracy);
            end
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            addpoints(lineClassificationLoss,iteration,double(gather(extractdata(classificationLoss))));
            title("Epoch: " + epoch + ", Elapsed: " + string(D))
            drawnow
        end
    end
end
accuracy = cvAccuracy(dlnet, cX, categorical(cY), miniBatchSize, executionEnvironment, 1)
function accuracy = cvAccuracy(dlnet, XTest, YTest, miniBatchSize, executionEnvironment, confusionChartFlg)
    dlXTest = dlarray(XTest,'SSCB');
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        dlXTest = gpuArray(dlXTest);
    end
    dlYPred = modelPredictions(dlnet,dlXTest,miniBatchSize);
    [~,idx] = max(extractdata(dlYPred),[],1);
    YPred = categorical(idx);
    accuracy = mean(YPred(:) == YTest(:));
    if confusionChartFlg == 1
        figure
        confusionchart(YPred(:),YTest(:));
    end
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
function [gradients,state,loss,classificationLoss] = modelGradientsOnWeights(dlnet,dlX,Y)
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
    
    classificationLoss = crossentropy(squeeze(dlYPred),Y) + scalarL2Factor*penalty;
    loss = classificationLoss;
%     loss = classificationLoss + 0.2*(max(max(rawPredictions))-min(max(rawPredictions)));
    gradients = dlgradient(loss,dlnet.Learnables);
    %gradients = dlgradient(loss,dlnet.Learnables(4,:));
end
function [params,velocityUpdates,momentumUpdate] = adamFunction(params, rawParamGradients,...
    velocities, learnRates, momentums, L2Foctors, gradientMasks, iters)
    % https://arxiv.org/pdf/2010.07468.pdf %%AdaBelief
    % https://arxiv.org/pdf/1711.05101.pdf  %%DeCoupled Weight Decay 
    b1 = 0.5; 
    b2 = 0.999;
    e = 1e-8;
    curIter = iters(:);
    curIter = curIter(1);
    
    gt = rawParamGradients;
    mt = (momentums.*b1 + ((1-b1)).*gt);
    vt = (velocities.*b2 + ((1-b2)).*((gt-mt).^2)) + e;
    momentumUpdate = mt;
    velocityUpdates = vt;
    h_mt = mt./(1-b1.^curIter);
    h_vt = vt./(1-b2.^curIter);
%%%%----------- Check Point 3:  
%%%% Here you can specify whether to use bias correction, 
%%%% or zero-bias dense layer 
%%%% in this test, we can just try to eliminate the effect of varying learning
%%%% rates
%    params = params - 0.001.*(mt./(sqrt(vt)+e)).*gradientMasks...
%        - wd.*params.*gradientMasks; %This works better for zero-bias dense layer
     params = params - learnRates.*(h_mt./(sqrt(h_vt)+e)).*gradientMasks...
         -2*learnRates .* L2Foctors.*params.*gradientMasks;
%%%%
%%%%-----------End of Check Point 3 
end
function param = sgdFunction(param,paramGradient)
    learnRate = 0.01;
    param = param - learnRate.*paramGradient;
end
function [params, velocityUpdates] = sgdmFunction(params, paramGradients,...
    velocities, learnRates, momentums)
% https://towardsdatascience.com/stochastic-gradient-descent-momentum-explanation-8548a1cd264e
%     velocityUpdates = momentums.*velocities+learnRates.*paramGradients;
    velocityUpdates = momentums.*velocities+0.001.*paramGradients;
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
