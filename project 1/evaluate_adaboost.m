function [pb_test, adaboost_output,n_missclassify_test, err_test_final] =  evaluate_adaboost(X,Y,alpha,coordinate_wl,s_polarity_wl,theta_wl)

nrounds = numel(alpha);
ndata = numel(Y);
n_missclassify_test = zeros(nrounds,1);
f = zeros(nrounds,ndata);

for it = 1:nrounds
    
    weak_learner_output = evaluate_stump(X,coordinate_wl(it),s_polarity_wl(it),theta_wl(it));
    f(it+1,:)   = f(it,:)   + alpha(it).*weak_learner_output;
    n_missclassify_test(it) = sum(Y~=(f(it+1,:) > 0)); 
end

err_test_final = n_missclassify_test(end) / ndata;
adaboost_output = (f(end,:) > 0);
pb_test = f(end,:);
end