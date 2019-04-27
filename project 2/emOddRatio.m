function r = emOddRatio(model1, model2, test)
    k1 = size(model1.mu, 2);
    k2 = size(model2.mu, 2);
    n = size(test, 2);
    
    p1 = zeros(n, 1);
    for i = 1:k1
        p1 = p1 + model1.pi(i) * mvnpdf(test', model1.mu(:, i)', model1.sigma(:, :, i));
    end   
    
    p2 = zeros(n, 1);
    for i = 1:k2
        p2 = p2 + model2.pi(i) * mvnpdf(test', model2.mu(:, i)', model2.sigma(:, :, i));
    end       
    
    r = p1 ./ p2;
end

