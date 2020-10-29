classdef yxSoftmax < nnet.layer.Layer

    properties (Learnable)
        % Layer learnable parameters
%         Weights,Biases;
    end
    methods
        function layer = yxSoftmax(name) 
            % Set layer name.
            layer.Name = name;
            % Set layer description.
            layer.Description = "Yongxin's Softmax for compatibility";
        end
        function Z = predict(layer, X)
            sX = exp(squeeze(X)); 
            Z = sX ./ sum(sX);
        end
    end
end
