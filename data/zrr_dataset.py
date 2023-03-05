import numpy as np
import os

import torch

from data.base_dataset import BaseDataset
from .imlib import imlib
from multiprocessing.dummy import Pool
from tqdm import tqdm
import glob
from torchvision.transforms import ToTensor
from PIL import Image
from util.util import augment, remove_black_level, get_coord
from util.util import extract_bayer_channels
import random
import cv2
root_path = '/home/lab611'

val_list = [109, 120, 142, 150, 189, 295, 710, 794, 912, 925, 674, 704, 712, 711, 743, 751, 800, 857, 968, 998]

# Zurich RAW to RGB (ZRR) dataset
class ZRRDataset(BaseDataset):

	def __init__(self, opt, split='train', dataset_name='ZRR'):
		super(ZRRDataset, self).__init__(opt, split, dataset_name)

		self.batch_size = opt.batch_size
		self.mode = opt.mode  # RGB, Y or L=
		self.imio = imlib(self.mode, lib=opt.imlib)
		self.raw_imio = imlib('RAW', fmt='HWC', lib='cv2')
		self.opt = opt


		if split == 'train':
			# self.raw_dir = os.path.join(self.root, 'train_my', '320_axis_mr')
			# self.dslr_dir = os.path.join(self.root, 'train_my', '640_flir_hr')
			# self.names = ['%s'%i for i in range(0, 46839)]  # 46839
			# self.names = sorted(
			# 	glob.glob(os.path.join(self.raw_dir, '*.jpg'))
			# )
			self.names_lr, self.names = self.get_img_list(attr='train')
			# print(self.names)
			self._getitem = self._getitem_train

		elif split == 'val':
			# self.raw_dir = os.path.join(self.root, 'validation_my', '320_axis_mr')
			# self.dslr_dir = os.path.join(self.root, 'validation_my', '640_flir_hr')
			# # self.names = ['%s'%i for i in range(0, 1204)]
			# self.names = sorted(
			# 	glob.glob(os.path.join(self.raw_dir, '*.jpg'))
			# )
			self.names_lr, self.names = self.get_img_list(attr='val')
			self._getitem = self._getitem_val

		elif split == 'test':
			# self.raw_dir = os.path.join(self.root, 'test', 'huawei_raw')
			# self.dslr_dir = os.path.join(self.root, 'test', 'canon')
			# self.names = ['%s'%i for i in range(0, 1204)]
			self.names_lr, self.names = self.get_img_list(attr='test')
			self._getitem = self._getitem_test

		elif split == 'visual':
			self.raw_dir = os.path.join(self.root, 'full_resolution/huawei_raw')
			self.names = ['1072', '1096', '1167']
			self._getitem = self._getitem_visual

		else:
			raise ValueError

		self.len_data = len(self.names)

		self.raw_images = [0] * self.len_data
		self.coord = get_coord(H=480, W=640, x=1, y=1)
		read_images(self)

	def __getitem__(self, index):
		return self._getitem(index)
	def get_img_list(self, attr):
		if attr == 'train':
			img_list_hr = list(glob.glob('{}/Data/PBVS/track1/challengedataset/train_my/640_flir_hr/*.jpg'.format(root_path)))
			img_list_hr = [img_name for img_name in img_list_hr if os.path.basename(img_name) not in val_list]
			img_list_lr = list(glob.glob('{}/Data/PBVS/track1/challengedataset/train_my/320_axis_mr/*.jpg'.format(root_path)))
			img_list_lr = [img_name for img_name in img_list_lr if os.path.basename(img_name) not in val_list]
		elif attr == 'val':
			img_list_hr = [os.path.join(f'{root_path}/Data/PBVS/track1/challengedataset/train_my/640_flir_hr/{str(img_name).zfill(4)}.jpg') for
						img_name in val_list]
			img_list_lr = [os.path.join(f'{root_path}/Data/PBVS/track1/challengedataset/train_my/320_axis_mr/{str(img_name).zfill(4)}.jpg') for
						   img_name in val_list]
		else:
			img_list_lr = glob.glob('/home/lab611/Data/PBVS/track1/challengedataset/testingSetInput/evaluation2/mr_realr/*.jpg'.format(root_path))
			img_list_hr = glob.glob('/home/lab611/Data/PBVS/track1/challengedataset/testingSetInput/evaluation2/mr_real/*.jpg'.format(root_path))
			# img_list_hr = [os.path.join(f'{root_path}/Data/PBVS/track1/challengedataset/train_my/640_flir_hr/{str(img_name).zfill(4)}.jpg') for
			# 			img_name in val_list]
			# img_list_lr = [os.path.join(f'{root_path}/Data/PBVS/track1/challengedataset/train_my/320_axis_mr/{str(img_name).zfill(4)}.jpg') for
			# 			   img_name in val_list]
		return img_list_lr, img_list_hr
	def __len__(self):
		return self.len_data

	def get_patch(self, *args, patch_size=192*2, scale=2, multi=False, input_large=False):
		ih, iw = args[0].shape[:2]

		if not input_large:
			p = scale if multi else 1
			tp = p * patch_size
			ip = tp // scale
		else:
			tp = patch_size
			ip = patch_size

		ix = random.randrange(0, iw - ip + 1)
		iy = random.randrange(0, ih - ip + 1)

		if not input_large:
			tx, ty = scale * ix, scale * iy
		else:
			tx, ty = ix, iy

		ret = [
				args[0][iy:iy + ip, ix:ix + ip, :],
				args[1][ty:ty + tp, tx:tx + tp, :],
			args[2][ty:ty + tp, tx:tx + tp, :],
			]

		return ret

	def _getitem_train(self, idx):
		# raw_combined, raw_demosaic = self.raw_images[idx], self.raw_images[idx]
		#self._process_raw(self.raw_images[idx])
		raw = Image.open(self.names_lr[idx])#self.imio.read(os.path.join(self.raw_dir, self.raw_images[idx]))
		raw = raw.resize((640, 480), resample=Image.BICUBIC)
		raw_demosaic = raw.resize((640,480), resample=Image.BICUBIC)
		# # raw_combined = raw_demosaic
		# (filepath, tempfilename) = os.path.split(os.path.join(self.raw_dir, self.names[idx]))
		# # print(tempfilename)
		# print(self.names_lr[idx], self.names[idx])
		dslr_image = Image.open(self.names[idx])#.convert('L')
		# dslr_image = Image.open(os.path.join(self.dslr_dir, self.raw_images[idx]))
		#self.imio.read(os.path.join(self.dslr_dir, self.names[idx]))
		dslr_image = np.array(dslr_image) #load(self.names[idx])
		if self.opt.add_mean == 1:
			raw_demosaic = np.array(raw_demosaic) + 18.550807  # load(self.names_lr[idx])
			raw = np.array(raw) + 18.550807
		else:
			raw_demosaic = np.array(raw_demosaic)  # load(self.names_lr[idx])
			raw = np.array(raw)
		# print("raw_demasaic: ", raw_demosaic.shape, " raw: ", raw.shape, " dslr: ", dslr_image.shape, self.coord.shape)
		coord = self.coord.transpose(1,2,0)
		# print("raw_demasaic1: ", raw_demosaic.shape, " raw1: ", raw.shape, " dslr1: ", dslr_image.shape, coord.shape)
		# raw_demosaic, raw, dslr_image = self.get_patch(raw_demosaic, raw, dslr_image)
		# print("raw: ", raw_demosaic.shape, dslr_image.shape, self.coord.shape)
		# raw_demosaic, raw_combined, dslr_image = augment(raw_demosaic, raw, dslr_image)
		raw_demosaic = ToTensor()(raw_demosaic)
		raw = ToTensor()(raw)
		dslr_image = ToTensor()(dslr_image)
		coord = ToTensor()(coord)
		raw = raw.type(torch.FloatTensor)
		raw_demosaic = raw_demosaic.type(torch.FloatTensor)
		# print("tensor: ", raw_demosaic.shape, dslr_image.shape, self.coord.shape)
		return {'raw': raw,
				'raw_demosaic': raw_demosaic,
				'dslr': dslr_image,
				'coord': coord,
				'fname': self.names[idx]}

	def _getitem_val(self, idx):
		raw = Image.open(self.names_lr[idx]) # self.imio.read(os.path.join(self.raw_dir, self.raw_images[idx]))
		# # raw_combined = raw_demosaic
		# (filepath, tempfilename) = os.path.split(os.path.join(self.raw_dir, self.names[idx]))
		# print(self.names_lr[idx], self.names[idx])
		raw = raw.resize((640, 480), resample=Image.BICUBIC)
		raw_demosaic = raw.resize((640, 480), resample=Image.BICUBIC)
		dslr_image = Image.open(self.names[idx])#.resize((640, 480), resample=Image.BICUBIC)  # .convert('L')
		dslr_image = np.array(dslr_image)# load(self.names[idx])
		if self.opt.add_mean == 1:
			raw_demosaic = np.array(raw_demosaic) + 18.550807  # load(self.names_lr[idx])
			raw = np.array(raw) + 18.550807
		else:
			raw_demosaic = np.array(raw_demosaic)  # load(self.names_lr[idx])
			raw = np.array(raw)
		coord = self.coord.transpose(1,2,0)
		raw = ToTensor()(raw)
		dslr_image = ToTensor()(dslr_image)
		raw_demosaic = ToTensor()(raw_demosaic)
		coord = ToTensor()(coord)
		raw = raw.type(torch.FloatTensor)
		raw_demosaic = raw_demosaic.type(torch.FloatTensor)
		return {'raw': raw,
				'raw_demosaic': raw_demosaic,
				'dslr': dslr_image,
				'coord': coord,
				'fname': self.names[idx]}
	def _getitem_test(self, idx):
		raw = Image.open(self.names[idx]) # self.imio.read(os.path.join(self.raw_dir, self.raw_images[idx]))
		# # raw_combined = raw_demosaic
		# (filepath, tempfilename) = os.path.split(os.path.join(self.raw_dir, self.names[idx]))
		# print(self.names_lr[idx], self.names[idx])
		raw = raw.resize((640, 480), resample=Image.BICUBIC)
		raw_demosaic = raw.resize((640, 480), resample=Image.BICUBIC)
		dslr_image = Image.open(self.names[idx]).resize((640, 480), resample=Image.BICUBIC)  # .convert('L')
		dslr_image = np.array(dslr_image)# load(self.names[idx])
		if self.opt.add_mean == 1:
			raw_demosaic = np.array(raw_demosaic) + 18.550807  # load(self.names_lr[idx])
			raw = np.array(raw) + 18.550807
		else:
			raw_demosaic = np.array(raw_demosaic)  # load(self.names_lr[idx])
			raw = np.array(raw)
		coord = self.coord.transpose(1,2,0)
		# print("raw_demasaic1: ", raw_demosaic.shape, " raw1: ", raw.shape, " dslr1: ", dslr_image.shape, coord.shape)
		raw = ToTensor()(raw)
		dslr_image = ToTensor()(dslr_image)
		raw_demosaic = ToTensor()(raw_demosaic)
		coord = ToTensor()(coord)
		raw = raw.type(torch.FloatTensor)
		raw_demosaic = raw_demosaic.type(torch.FloatTensor)
		return {'raw': raw,
				'raw_demosaic': raw_demosaic,
				'dslr': dslr_image,
				'coord': coord,
				'fname': self.names[idx]}
	# def _getitem_visual(self, idx):
	# 	raw_combined, raw_demosaic = self._process_raw(self.raw_images[idx])
	# 	h, w = raw_demosaic.shape[-2:]
	# 	coord = get_coord(H=h, W=w, x=1, y=1)
	#
	# 	return {'raw': raw_combined,
	# 			'raw_demosaic': raw_demosaic,
	# 			'dslr': raw_combined,
	# 			'coord': self.coord,
	# 			'fname': self.names[idx]}

	# def _process_raw(self, raw):
	# 	raw = remove_black_level(raw)
	# 	raw_combined = extract_bayer_channels(raw)
	# 	raw_demosaic = get_raw_demosaic(raw)
	# 	return raw_combined, raw_demosaic

def iter_obj(num, objs):
	for i in range(num):
		yield (i, objs)

def imreader(arg):
	# Due to the memory (32 GB) limitation, here we only preload the raw images. 
	# If you have enough memory, you can also modify the code to preload the sRGB images to speed up the training process.
	i, obj = arg
	for _ in range(3):
		try:
			obj.raw_images[i] = obj.raw_imio.read(os.path.join(obj.raw_dir, obj.names[i]))
			failed = False
			break
		except:
			failed = True
	if failed: print('%s fails!' % obj.names[i])

def read_images(obj):
	# may use `from multiprocessing import Pool` instead, but less efficient and
	# NOTE: `multiprocessing.Pool` will duplicate given object for each process.
	print('Starting to load images via multiple imreaders')
	pool = Pool() # use all threads by default
	for _ in tqdm(pool.imap(imreader, iter_obj(obj.len_data, obj)), total=obj.len_data):
		pass
	pool.close()
	pool.join()

if __name__ == '__main__':
	pass
