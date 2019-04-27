close all;
clc;
path('Dataset', path);
path('Gradients', path);
path('Textons', path);
path('Filters', path);
path('Util', path);
path('../boosting', path)
%%
%read ground truth segmentation
M = readSeg('101085.seg');
M1 = seg2bmap(M,size(M,2),size(M,1));
imshow(~M1,[]); 
SE = strel('disk',4);
M1 = imdilate(M1,SE);
figure,imshow(~imdilate(M1,SE));
%%
im = (double(imread('101085.jpg'))/255);
imshow(~M1,[]);
figure, imshow(~edge(rgb2gray(im),'canny'));
%%
im =  max(min(imresize(im,.3),1),0);
M1 = logical(round(imresize(double(M1), size(im, 1) / size(M1, 1))));
%%

%change method to 2 for adaboost

method = 1;
make_balanced = 0;
ndata = 500;
[y_train,f_train,y_test,f_test,y,f_all] = prepare_data(im,M1,make_balanced,ndata);

f_mean =mean(f_all, 2);
comp = pcacov(cov(f_all'));
proj_train = comp * (f_train - repmat(f_mean, 1, size(f_train, 2)));
proj_test = comp * (f_test - repmat(f_mean, 1, size(f_test, 2)));
figure,
plot(proj_train(1,:), proj_train(2,:), '.g'), hold on
plot(proj_test(1,:), proj_test(2,:), '.r')



if method ==1
     % preprocess data
     fstd = std(f_train, 0, 2);
     fstd = fstd + (fstd==0);
     fmean = sum(f_train, 2) / size(f_train, 2);
     f_train = (f_train - repmat(fmean,1,size(f_train,2))) ./ repmat(fstd,1,size(f_train,2));
     f_test = (f_test - repmat(fmean,1,size(f_test,2))) ./ repmat(fstd,1,size(f_test,2));
     f_all = (f_all - repmat(fmean,1,size(f_all,2))) ./ repmat(fstd,1,size(f_all,2));
    
     f_train = [f_train; ones(1, size(f_train, 2))];
     f_test = [f_test; ones(1, size(f_test, 2))];
     f_all = [f_all; ones(1, size(f_all, 2))];
     
     % train logistic regression
    [beta,n_missclassify_train,err_train_logistic] = logistic_regression(f_train',y_train');
else
    nrounds = 500;
    [alpha,coordinate_wl,s_polarity_wl,theta_wl,f_final,n_missclassify_train,err_train_adaboost] = adaboost(f_train,y_train,nrounds);
end
        
if method ==1
    pb_test = 1 ./ (1 + exp(-f_test'*beta));
    pb_all = 1 ./ (1 + exp(-f_all'*beta));    
    n_missclassify_test = sum(repmat(y_test, size(beta, 2), 1)'~=(pb_test>0.5));
    err_test_logistic = n_missclassify_test/ numel(y_test);
    n_missclassify_all = sum(repmat(y, size(beta, 2), 1)'~=(pb_all>0.5));
    err_all_logistic = n_missclassify_all/numel(y);
    
    figure,
    plot(err_train_logistic, 'g'), hold on
    plot(err_test_logistic, 'r'), hold on
    
else
    [pb_test, adaboost_output_test,n_missclassify_test, err_test_final] =  evaluate_adaboost(f_test,y_test,alpha,coordinate_wl,s_polarity_wl,theta_wl);
    [pb_all, adaboost_output_all ,n_missclassify_all, err_all_final] =  evaluate_adaboost(f_all,y,alpha,coordinate_wl,s_polarity_wl,theta_wl);
    pb_test = 1./(1+exp(-2*pb_test));
    pb_all = 1./(1+exp(-2*pb_all));    
    err_test_adaboost = n_missclassify_test / numel(y_test);
    err_all_adaboost = n_missclassify_all/numel(y);  
    
    figure,
    plot(err_train_adaboost, 'g'), hold on
    plot(err_test_adaboost, 'r'), hold on    
end

figure,
plot(err_train_adaboost, 'g'), hold on
plot(err_train_logistic, 'r'), hold on  

clf,
plot(err_test_adaboost, 'g'), hold on
plot(err_test_logistic, 'r'), hold on  


edges = double(pb_all > 0.5);
%zs(idxs_nonzero) = pb;
result = edge((rgb2gray(im)),'canny');
softedges = zeros(size(result));
figure,imshow(~result);
[idx_horizontal,idx_vertical]  = find(result==1);

for i=1:numel(idx_horizontal)
          id_x = idx_horizontal(i);
          id_y = idx_vertical(i);
          result(id_x,id_y)=edges(i);
          softedges(id_x,id_y) = pb_all(i);
end
figure,imshow(~result);
figure,imshow(1-softedges);


