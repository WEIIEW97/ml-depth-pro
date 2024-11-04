clc, clear;

% addpath('/home/william/Codes/ml-depth-pro/build');
% 
% onnx_path='/home/william/Codes/ml-depth-pro/onnx_exp/depth_pro.onnx';
% img_path='/home/william/Codes/ml-depth-pro/data/example.jpg';

% addpath('D:\william\codes\ml-depth-pro\build\Debug');
% addpath('D:\william\codes\ml-depth-pro\csrc\mex');

onnx_path='D:\william\codes\ml-depth-pro\onnx_exp\depth_pro.onnx';
img_path='D:\william\codes\ml-depth-pro\data\example.jpg';
num_cpu_thread=4;
f_px=0;
verbose=1;
use_cuda=1;

[inverse_depth, f_px] = mex_depth_pro_onnx_infer(onnx_path, img_path, num_cpu_thread, f_px, verbose, use_cuda);

