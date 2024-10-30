function net = load_onnx(onnx_path)
%LOAD_ONNX Summary of this function goes here
%   Detailed explanation goes here

% restriction
% Warning: The ONNX file uses IR version 9, while the highest fully-supported IR is version 7. 
% Warning: The ONNX file uses Opset version 19, while the highest fully-supported version is 14. 
% The imported network may differ from the ONNX network. 
net=importNetworkFromONNX(onnx_path);
end

