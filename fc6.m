clc;clear;close;
tic;

%% confrim caffe path
% Add caffe/matlab to your Matlab search PATH in order to use matcaffe
% if exist('C:\caffe-windows\Build\x64\Release\matcaffe\+caffe', 'dir')
%     addpath('C:\caffe-windows\Build\x64\Release\matcaffe');
% else
%     error('Please run this demo from caffe/matlab/demo');
% end

% Set caffe mode
caffe.set_mode_gpu();
gpu_id = 0; % 1 to use the GPU, 0 to use the CPU
caffe.set_device(gpu_id);

%% path setting
% image path setting
image_path = 'images/';

% model path setting
phase = 'test';
path = './';
mean_path='./';
txt_path='./';
net_model = [path, 'deploy.prototxt'];
net_weights = [path, 'model/_iter_480.caffemodel'];
net = caffe.Net(net_model, net_weights, phase); %Initialize the network

% load image mean
image_mean = caffe.io.read_mean([path,'mean.binaryproto']); % image_mean is already in W x H x C with BGR channels

% text file
test_phase = {'val', 'test'};
mae_phase = {'classification', 'regression'};

%% main code
for test_phase_num = 2:2%size(test_phase, 2)
    
    [all_image_path_name, all_label, all_sigma] = textread([txt_path,test_phase{test_phase_num},'.txt'], '%s %f %f');
    
    counter = 0;
    save_all_image_path_name = {};
    all_original_scores = [];
    all_normalization_scores = [];
    for num = 1:size(all_image_path_name, 1)
        
        %%% check stage
        image_path_name = cell2mat(all_image_path_name(num));
        
        %%% information of image
        counter = counter + 1;
        label = all_label(num);
        result(counter, 1) = label; %real label
        
        %%% image preprocessing
        % input_data is Height x Width x Channel x Num
        image = imread([image_path,image_path_name]); % color image as uint8 HxWx3
%         image = rgb2gray(image);
        image = permute(image, [2, 1, 3, 4]); % flip width and height to make width the fastest dimension -> [width, height, channels, images]
        image = single(image); % convert from uint8 to single
        image = imresize(image, [64, 64], 'bilinear'); % test image resize to train image sizes
        image_mean = imresize(image_mean, [64, 64], 'bilinear'); % test image resize to train image sizes
        im_data = image - image_mean; % subtract mean_data (already in W x H x C with BGR channels)
        crops_data = zeros(64, 64, 1, 1, 'single');
        crops_data(:, :, :, 1) = im_data(:, :, :)/1;
        
        %%% do forward pass to get scores
        % The net forward function. It takes in a cell array of N-D arrays
        % (where N == 4 here) containing data of input blob(s) and outputs a cell
        % array containing data from output blob(s)
        scores = net.forward({crops_data}); % scores are now Channels x Num
        scores = net.blobs('fc6').get_data();
        scores(scores<(0.6*max(scores)))=0;
        plot(scores);
        saveas(gcf, ['fc_max/',image_path_name]);
%         scores = scores{1};
%         sum_scores = sum(scores);
%         nor_scores = scores/sum_scores;
%         [~, maxlabel] = max(scores); % the label of the highest score
%         predict_age = maxlabel;
        
        %%% record the result
%         result(counter, 2) = maxlabel-1; %predict label
        
        %%% store variable
%         save_all_image_path_name{counter, 1} = image_path_name;
%         all_original_scores = [all_original_scores, scores];
%         all_normalization_scores = [all_normalization_scores, nor_scores];
        
    end
    
    %%% save final result
end
