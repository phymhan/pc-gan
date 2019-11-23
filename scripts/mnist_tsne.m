src = '../sourcefiles/MNIST_sample.txt';

f = readNPY('~/mnist-image.npy');
l = readNPY('~/mnist-label.npy');
t = readNPY('~/mnist-thick.npy');

f = reshape(f, 1000, 784);
y = tsne(f);

shapes = {'o', '+', '*', 'v', 'x', 's', 'd', '^', 'p', 'h'};
colors = {[0    0.4470    0.7410], [0.8500    0.3250    0.0980], [0.9290    0.6940    0.1250]};

hf = figure;
hold on
for i = 1:3
    plot(nan, nan, 's', 'markerfacecolor', colors{i}, 'markeredgecolor', colors{i})
end
hold off

for i = 1:length(l)
    line(y(i,1), y(i,2), 'Marker', shapes{l(i)+1}, 'MarkerSize', 4, 'Color', colors{t(i)});
end
legend('thin', 'normal', 'thick')

hf.Position = [793 275 250 235];
grid on
box on
set(hf, 'color', [1 1 1])
hf.Position = [793 275 250 235];

% export_fig mnist_tsne.pdf
