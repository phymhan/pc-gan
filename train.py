import time
import math
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    total_iter = 0
    num_iter_per_epoch = math.ceil(dataset_size / opt.batchSize)
    opt.num_iter_per_epoch = num_iter_per_epoch
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()  # reset() moved here

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_iter % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            # visualizer.reset()  # TODO: in original code, why reset() before every iteration?
            model.set_input(data)
            model.optimize_parameters()
            total_iter += 1
            epoch_iter += 1

            if total_iter % opt.display_freq == 0:
                save_result = total_iter % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iter, save_result)

            if total_iter % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch - 1, epoch_iter / num_iter_per_epoch, opt, losses)

            if total_iter % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iter %d)' % (epoch, total_iter))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iter))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
