function [alpha,coordinate_wl,s_polarity_wl,theta_wl,f,n_missclassify_train,err_train_adaboost] = adaboost(X,Y,nrounds)

Y = 2 * Y - 1;

nclass_1 = size(X,2);
ndata = length(Y);

Distribution_on_indexes = ones(1,ndata)/ndata; 

decision = 0;

n_missclassify_train = zeros(nrounds,1);
f = zeros(nrounds,ndata);

% figure
for it = 1:nrounds,

    %%--------------------------------------------------------
    %% Find best weak learner at current round of boosting
    %%--------------------------------------------------------
    
    %%
    %% consider all decision stump classifiers
    %% four cases
    %% 1) two possible coordinates
    %% 2) two possible polarities ('s_polarity') (1 if less/more than threshold)
    %% and all possible thresholds 
    
    for coordinate = 1:size(X,1)
        input_dim  = X(coordinate,:);
        %%------------------------------------------------------
        %% try  all possible thresholds and see for which one
        %% it is minimized
        %%------------------------------------------------------

        for threshold_index = [1:length(input_dim)],

            candidate_threshold =  input_dim(threshold_index);
            s_index = 0;
            for s_polarity = [1,-1],
                s_index = s_index + 1;

                candidate_learner_output =  evaluate_stump(X,coordinate,s_polarity,candidate_threshold);         
                %%  evaluate classifier error
                
                error_for_this_combination = (Distribution_on_indexes * (candidate_learner_output ~= Y)') / sum(Distribution_on_indexes);

                error(coordinate,s_index,threshold_index) = error_for_this_combination;
            end
        end
    end

    %% we found minimum, use its coordinates to determine best weak learner    
    error_min =  min(error(:));    
    idx = find(error == error_min);
    [coordinate_wl(it),s_index,thr_index] = ind2sub(size(error),idx(1));
    theta_wl(it)      = X(coordinate_wl(it),thr_index);
    s_polarity_wl(it) = (s_index == 1)*(1) + (s_index == 2)*(-1);

    weak_learner_output = evaluate_stump(X,coordinate_wl(it),s_polarity_wl(it),theta_wl(it));    
  
    %%--------------------------------------------------------
    %% Incorporate current weak learner into the rest of the classifier
    %%--------------------------------------------------------
   
    
    alpha(it) = log((1 - error_min) / error_min) / 2;
    Distribution_on_indexes = Distribution_on_indexes .* exp(-alpha(it) * Y .* weak_learner_output);
    Distribution_on_indexes = Distribution_on_indexes ./ sum(Distribution_on_indexes);


    f(it+1,:)   = f(it,:)   + alpha(it).*weak_learner_output;
    n_missclassify_train(it) = sum(Y~=(1 * (f(it+1,:) > 0) + (-1) * (f(it+1,:) < 0)));

end


err_train_adaboost = n_missclassify_train(end) / ndata;
f = f(2:end,:);


end