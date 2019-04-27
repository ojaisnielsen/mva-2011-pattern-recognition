function [beta,n_missclassify_train,err_train_logistic] = logistic_regression(X,Y)
w_logistic = zeros(size(X, 2), 1);
n_missclassify_train = [];
beta = [];

cnt = 0; crit = [];
while 1==1,
    %% compute the probability P(y=1|X) in terms of g

    g_values = 1 ./ (1 + exp(-X * w_logistic));

    cnt = cnt+1;

    crit(cnt) = sum(Y .* log(g_values) + (1 - Y) .* log(1 - g_values));
    
    %%
    % Jacobian  of criterion we want to minimize
%     for dim_1 = 1:dim_data
%         Jacobian(dim_data,1) = yout_code_here;        
%     end
    Jacobian =  X' * (Y - g_values); 

    % Hessian of criterion we want to minimize

%     Hessian = zeros(dim_data,dim_data);
%     for dim_1 = 1:dim_data
%         for dim_2 = 1:dim_data
%             Hessian(dim_1,dim_2) = your_code_here;
%         end
%     end
    R = diag(g_values .* (1 - g_values));
    Hessian = -X' * R * X;
    change  = - inv(Hessian)*Jacobian;
    w_logistic = w_logistic + change;
    pb_train = 1 ./ (1 + exp(-X*w_logistic));
    
    
    n_missclassify_train(cnt) = sum(Y~=(pb_train>0.5));
    beta = [beta, w_logistic];

    if (abs(det(Hessian))<10e-10)|(max(abs(change)) < 10e-10)
      break; 
    end;
end


err_train_logistic = n_missclassify_train/numel(Y);

end