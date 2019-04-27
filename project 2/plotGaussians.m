function plotGaussians(mu, sigma, nPoints)

    n = size(mu, 2);
    styles = {'r'; 'g'; 'b'; 'k'};
    
    if (nargin == 2)
        nPoints = 100;
    end
    
    for i = 1:n
        style = mod(i - 1, numel(styles)) + 1;
        [~, S, V] = svd(sigma(:,:,i));
        points = V'*([2 * sqrt(S(1, 1)) * cos(2 * pi * (1:nPoints) / (nPoints - 1)); 2 * sqrt(S(2, 2)) * sin(2 * pi * (1:nPoints) / (nPoints - 1))]);
        points = points' + ones(nPoints, 1) * mu(:, i)';
        plot(points(:, 1), points(:, 2), styles{style}), hold on   
    end

end

