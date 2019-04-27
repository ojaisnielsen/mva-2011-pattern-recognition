function plotClusters(data, assignments)
    styles = {'.r'; '.g'; '.b'; '.k'};
    for n = 1:size(data, 2),
        style = mod(assignments(n) - 1, numel(styles)) + 1;
        scatter(data(1, n), data(2, n), styles{style}), hold on
    end
end

