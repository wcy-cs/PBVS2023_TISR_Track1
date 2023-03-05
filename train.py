#-*- encoding: UTF-8 -*-
# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")
# import random
#
# def set_random_seed(seed):
#     """Set random seeds."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import numpy as np
import math
import sys
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp

from util.util import calc_psnr as calc_psnr
from util.util import calc_psnr1
from util.util import torch_psnr

if __name__ == '__main__':
	opt = TrainOptions().parse()
	# set_random_seed(opt.seed)
	dataset_train = create_dataset(opt.dataset_name, 'train', opt)

	dataset_size_train = len(dataset_train)
	print('The number of training images = %d' % dataset_size_train)
	dataset_val = create_dataset(opt.dataset_name, 'val', opt)
	dataset_size_val = len(dataset_val)
	print('The number of val images = %d' % dataset_size_val)
	writer = SummaryWriter('./logs/{}'.format(opt.writer_name))
	model = create_model(opt)
	model.setup(opt)
	visualizer = Visualizer(opt)
	total_iters = ((model.start_epoch * (dataset_size_train // opt.batch_size)) \
					// opt.print_freq) * opt.print_freq

	for epoch in range(model.start_epoch + 1, opt.niter + opt.niter_decay + 1):
		# training
		epoch_start_time = time.time()
		epoch_iter = 0
		model.train()

		iter_data_time = iter_start_time = time.time()
		for i, data in enumerate(dataset_train):
			if total_iters % opt.print_freq == 0:
				t_data = time.time() - iter_data_time
			total_iters += 1 #opt.batch_size
			epoch_iter += 1 #opt.batch_size
			model.set_input(data)
			model.optimize_parameters()

			if total_iters % opt.print_freq == 0:
				losses = model.get_current_losses()
				t_comp = (time.time() - iter_start_time)
				visualizer.print_current_losses(
					epoch, epoch_iter, losses, t_comp, t_data, total_iters)
				if opt.save_imgs: # Too many images
					visualizer.display_current_results(
					'train', model.get_current_visuals(), total_iters)
				iter_start_time = time.time()

			iter_data_time = time.time()
		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d'
				  % (epoch, total_iters))
			model.save_networks(epoch)

		print('End of epoch %d / %d \t Time Taken: %.3f sec'
			  % (epoch, opt.niter + opt.niter_decay,
				 time.time() - epoch_start_time))
		model.update_learning_rate()
		# optimizer.param_groups[0]["lr"]
		writer.add_scalar('lr', model.optimizer_LiteISPNet.param_groups[0]["lr"], epoch)
		# val
		if opt.calc_metrics:
			model.eval()
			val_iter_time = time.time()
			tqdm_val = tqdm(dataset_val)
			psnr = [0.0] * dataset_size_val
			psnr1 = [0.0] * dataset_size_val
			psnr2 = [0.0] * dataset_size_val
			time_val = 0
			for i, data in enumerate(tqdm_val):
				model.set_input(data)
				time_val_start = time.time()
				with torch.no_grad():
					model.test()
				time_val += time.time() - time_val_start
				res = model.get_current_visuals()
				psnr[i] = calc_psnr(res['dslr_warp'], res['data_out'])#data['dslr'].cuda())#
				psnr1[i] = calc_psnr(data['dslr'].cuda(), res['data_out'])
				# psnr2[i] = calc_psnr1(data['dslr'].cuda(), res['data_out'])  # .item()
				# if opt.save_imgs:
				#     visualizer.display_current_results('val', res, epoch)
			visualizer.print_psnr(epoch, opt.niter + opt.niter_decay, time_val, np.mean(psnr))
			writer.add_scalar('val_psnr_warp', np.mean(psnr), epoch)
			writer.add_scalar('val_psnr_gt', np.mean(psnr1), epoch)
			# writer.add_scalar('val_psnr_rg', np.mean(psnr1), epoch)
			#visualizer.print_psnr(epoch, opt.niter + opt.niter_decay, time_val, np.mean(psnr1))
			# visualizer.writer.add_scalar('val/psnr', np.mean(psnr), epoch)
			# print('End of epoch %d / %d (Val) \t Time Taken: %.3f s \t PSNR: %f'
			#     % (epoch, opt.niter + opt.niter_decay, time_val, np.mean(psnr)))

		sys.stdout.flush()
