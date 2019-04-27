function model = em(data, k)

    [d, n] = size(data);

    % K-means

    nRounds = 20;

    minCost = Inf;
    bestCenters = [];
    bestAssignments = [];

    for round = 1:nRounds

        perm = randperm(n);
        centers = data(:, perm(1:k));

        converged = false;
        assignments = zeros(n, 1);

        cost = [];
        nIter = 0;

        while ~converged
            nIter = nIter + 1;
            converged = true;
            clusterSums = zeros(d, k);
            clusterSizes = zeros(1, k);
            for i = 1:n
                dists = zeros(k, 1);
                for c = 1:k,
                    dists(c) = norm(data(:, i) - centers(:, c));
                end
                [~, c] = min(dists);
                converged = converged && (assignments(i) == c);
                assignments(i) = c;
                clusterSums(:, c) = clusterSums(:, c) + data(:, i);
                clusterSizes(c) = clusterSizes(c) + 1;
            end

            for c = 1:k
                centers = clusterSums ./ repmat(clusterSizes, d, 1);
            end

            cost(nIter) = 0;
            for i = 1:n
                cost(nIter) = cost(nIter) + sum(data(:, i) - centers(:, assignments(i)) .^ 2);
            end        
        end        


        if (cost(end) < minCost(end))
            minCost = cost;
            bestCenters = centers;
            bestAssignments = assignments;
        end

    end

    figure, plot(minCost)
    title 'best K-means distortion evolution';
    figure, plotClusters(data, bestAssignments)
    title 'Best K-means clustering';

    % EM

    pi = zeros(1, k);
    mu = bestCenters;
    sigma = repmat(zeros(d), [1, 1, k]);
    for j = 1:k
        indices = find((bestAssignments == j))';
        pi(j) = numel(indices) / n;
        for i = indices
            centeredData = data(:, i) - mu(:, j);
            sigma(:,:,j) = sigma(:,:,j) + (centeredData * centeredData');
        end
        sigma(:,:,j) = sigma(:,:,j) / numel(indices);
    end

    figure, plotClusters(data, bestAssignments), hold on
    plotGaussians(mu, sigma)
    title 'EM initialization';

    R = zeros(n, k);
    loglik = [];
    converged = false;
    nIter = 0;
    while ~converged
        nIter = nIter + 1;

        % E step

        for j = 1:k
            R(:,j) = pi(j) * mvnpdf(data', mu(:, j)', sigma(:, :, j));
        end        
        R = R ./ repmat(sum(R, 2), 1, k);

        % M step

        for j = 1:k
            mu(:, j) = sum(repmat(R(:, j), 1, d) .* data')' / sum(R(:, j));            
            sigma(:,:,j) = zeros(d, d);
            for i = 1:n
                centeredData = data(:, i) - mu(:, j);
                sigma(:,:,j) = sigma(:,:,j) + R(i, j) * (centeredData * centeredData');
            end
            sigma(:,:,j) = sigma(:,:,j) / sum(R(:, j));
        end

        pi = sum(R) / n;

        loglik(nIter) = 0;
        for i = 1:n
            sumP = 0;
            for j = 1:k
                sumP = sumP + pi(j) * mvnpdf(data(:, i)', mu(:, j)', sigma(:, :, j));
            end
            loglik(nIter) = loglik(nIter) + log(sumP);
        end
        if nIter > 1
            converged = (loglik(nIter) - loglik(nIter - 1))^2 < 10^(-6);
        end
    end

    figure, plot(loglik)
    title 'EM loglikelihood evolution';
    
    figure, plotClusters(data, bestAssignments), hold on
    plotGaussians(mu, sigma)
    title 'EM final';
    
    model.pi = pi;
    model.mu = mu;
    model.sigma = sigma;

end

