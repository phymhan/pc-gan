function visualize_embedding(expr, p, ckpt_dir)
if ~exist('ckpt_dir', 'var')
    ckpt_dir = '../checkpoints_elo';
end
if ~exist('expr', 'var')
    expr = 'elo_UTK_bnn';
end
if ~exist('p', 'var') || isempty(p)
    p = 'f';
    if ~isempty(dir(fullfile(ckpt_dir, expr, 'stds*')))
        p = [p 's'];
    end
    if ~isempty(dir(fullfile(ckpt_dir, expr, 'vars*')))
        p = [p 'v'];
    end
    if ~isempty(dir(fullfile(ckpt_dir, expr, 'stds*'))) && ~isempty(dir(fullfile(ckpt_dir, expr, 'vars*')))
        p = [p 'vs'];
    end
end
dd = dir(fullfile(ckpt_dir, expr, 'labels*'));
epochs = sort(cellfun(@(s, idx) str2num(s(idx(1):idx(end))), {dd.name}, ...
    cellfun(@(s) regexp(s, '\d'), {dd.name}, 'UniformOutput', 0)));
if contains(p, 'f')
    c1s = [];
    c2s = [];
end
start_fig_number = get(gcf, 'Number');
if start_fig_number > 1
    start_fig_number = start_fig_number + 10;
end
for ep = epochs
    if contains(p, 'f')
        [c1, c2] = plot_f(ep);
        c1s(end+1) = c1;
        c2s(end+1) = c2;
    end
    if contains(p, 's')
        plot_s(ep);
    end
    if contains(p, 'v')
        plot_v(ep);
    end
    if contains(p, 'vs')
        plot_vs(ep);
    end
    pause(0.2)
end
if contains(p, 'f') && ~isempty(c1s)
    figure(start_fig_number+9);
    plot(epochs, c1s, epochs, c2s);
    title('Correlation')
    xlabel('epochs')
    ylabel('Correlations')
    legend('Spearman', 'Pearson', 'location', 'southeast')
    fprintf('max Spearman: %.4f, max Pearson: %.4f\n', max(c1s), max(c2s))
    fprintf('last Spearman: %.4f, last Pearson: %.4f\n', c1s(end), c2s(end))
end


    function f = read_npy(s, i)
        f = readNPY(fullfile(ckpt_dir, expr, sprintf('%s_%d.npy', s, i)));
    end

    function [c1, c2] = plot_f(i)
        l = read_npy('labels', i);
        f = read_npy('features', i);
        figure(start_fig_number+0);
        set(1, 'position', [100 500 550 450]);
        plot(l, f, '.');
        title(sprintf('feature (epoch %d), Spearman %.2f, Pearson %.2f', ...
            i, corr(l, f, 'type', 'Spearman'), corr(l, f, 'type', 'Pearson')))
        c1 = corr(l, f, 'type', 'Spearman');
        c2 = corr(l, f, 'type', 'Pearson');
    end

    function plot_s(i)
        l = read_npy('labels', i);
        s = read_npy('stds', i);
        figure(start_fig_number+1);
        set(2, 'position', [675 500 550 450]);
        plot(l, s, '.');
        title(sprintf('std (epoch %d)', i))
    end

    function plot_v(i)
        l = read_npy('labels', i);
        v = read_npy('vars', i);
        figure(start_fig_number+2);
        set(3, 'position', [1250 500 550 450])
        plot(l, sqrt(v), '.');
        title(sprintf('empirical std (epoch %d)', i))
    end

    function plot_vs(i)
        l = read_npy('labels', i);
        v = read_npy('vars', i);
        s = read_npy('stds', i);
        vs = sqrt(v + s.^2);
        figure(start_fig_number+3);
        set(4, 'position', [675 0 550 450]);
        plot(l, vs, '.');
        title(sprintf('var+std^2 (epoch %d)', i))
    end
end
