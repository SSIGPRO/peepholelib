# torch stuff
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from torch_nlm import nlm2d

def bit_depth_torch(x, bits):
    precisions = 2**bits
    return reduce_precision_torch(x, precisions)

def reduce_precision_torch(x: torch.Tensor,
                           npp: int,
                           min_val: float = None,
                           max_val: float = None) -> torch.Tensor:
    """
    Reduce precision of tensor x to npp uniform levels over the range [min_val, max_val].
    If min_val or max_val is None, they're inferred as x.min() and x.max().

    Args:
      x       : input tensor
      npp     : number of precision levels
      min_val : lower bound of range (float) or None to infer
      max_val : upper bound of range (float) or None to infer

    Returns:
      quantized tensor same shape as x
    """
    # 1) infer or use provided bounds
    if min_val is None:
        min_val = float(x.min())
    if max_val is None:
        max_val = float(x.max())
    # evita divisione per zero
    if max_val == min_val:
        return x.clone()

    # 2) shift & scale to [0,1]
    x01 = (x - min_val) / (max_val - min_val)

    # 3) quantize in [0,1]
    levels = npp - 1
    x_int = torch.round(x01 * levels)
    x_q01 = x_int / float(levels)

    # 4) rimappa in [min_val, max_val]
    x_q = x_q01 * (max_val - min_val) + min_val
    return x_q

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    https://gist.github.com/rwightman/f2d3849281624be7c0f11c85c87c1598
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
    
## Mapping between RGB and LAB color spaces
# The following functions are based on the code from: https://github.com/smartcameras/EdgeFool/blob/master/Train/rgb_lab_formulation_pytorch.py
def preprocess_lab(lab):
		L_chan, a_chan, b_chan =torch.unbind(lab,dim=2)
		# L_chan: black and white with input range [0, 100]
		# a_chan/b_chan: color channels with input range ~[-110, 110], not exact
		# [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
		return [L_chan / 50.0 - 1.0, a_chan / 110.0, b_chan / 110.0]


def deprocess_lab(L_chan, a_chan, b_chan):
		#TODO This is axis=3 instead of axis=2 when deprocessing batch of images 
			   # ( we process individual images but deprocess batches)
		#return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)
		return torch.stack([(L_chan + 1) / 2.0 * 100.0, a_chan * 110.0, b_chan * 110.0], dim=2)


def rgb_to_lab(srgb):

	srgb_pixels = torch.reshape(srgb, [-1, 3])

	linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).cuda()
	exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).cuda()
	rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
	
	rgb_to_xyz = torch.tensor([
				#    X        Y          Z
				[0.412453, 0.212671, 0.019334], # R
				[0.357580, 0.715160, 0.119193], # G
				[0.180423, 0.072169, 0.950227], # B
			]).type(torch.FloatTensor).cuda()
	
	xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)
	

	# XYZ to Lab
	xyz_normalized_pixels = torch.mul(xyz_pixels, torch.tensor([1/0.950456, 1.0, 1/1.088754]).type(torch.FloatTensor).cuda())

	epsilon = 6.0/29.0

	linear_mask = (xyz_normalized_pixels <= (epsilon**3)).type(torch.FloatTensor).cuda()

	exponential_mask = (xyz_normalized_pixels > (epsilon**3)).type(torch.FloatTensor).cuda()

	fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4.0/29.0) * linear_mask + ((xyz_normalized_pixels+0.000001) ** (1.0/3.0)) * exponential_mask
	# convert to lab
	fxfyfz_to_lab = torch.tensor([
		#  l       a       b
		[  0.0,  500.0,    0.0], # fx
		[116.0, -500.0,  200.0], # fy
		[  0.0,    0.0, -200.0], # fz
	]).type(torch.FloatTensor).cuda()
	lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor).cuda()
	#return tf.reshape(lab_pixels, tf.shape(srgb))
	return torch.reshape(lab_pixels, srgb.shape)

def lab_to_rgb(lab):
		lab_pixels = torch.reshape(lab, [-1, 3])
		# convert to fxfyfz
		lab_to_fxfyfz = torch.tensor([
			#   fx      fy        fz
			[1/116.0, 1/116.0,  1/116.0], # l
			[1/500.0,     0.0,      0.0], # a
			[    0.0,     0.0, -1/200.0], # b
		]).type(torch.FloatTensor).cuda()
		fxfyfz_pixels = torch.mm(lab_pixels + torch.tensor([16.0, 0.0, 0.0]).type(torch.FloatTensor).cuda(), lab_to_fxfyfz)

		# convert to xyz
		epsilon = 6.0/29.0
		linear_mask = (fxfyfz_pixels <= epsilon).type(torch.FloatTensor).cuda()
		exponential_mask = (fxfyfz_pixels > epsilon).type(torch.FloatTensor).cuda()


		xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29.0)) * linear_mask + ((fxfyfz_pixels+0.000001) ** 3) * exponential_mask

		# denormalize for D65 white point
		xyz_pixels = torch.mul(xyz_pixels, torch.tensor([0.950456, 1.0, 1.088754]).type(torch.FloatTensor).cuda())


		xyz_to_rgb = torch.tensor([
			#     r           g          b
			[ 3.2404542, -0.9692660,  0.0556434], # x
			[-1.5371385,  1.8760108, -0.2040259], # y
			[-0.4985314,  0.0415560,  1.0572252], # z
		]).type(torch.FloatTensor).cuda()

		rgb_pixels =  torch.mm(xyz_pixels, xyz_to_rgb)
		# avoid a slightly negative number messing up the conversion
		#clip
		rgb_pixels[rgb_pixels > 1] = 1
		rgb_pixels[rgb_pixels < 0] = 0

		linear_mask = (rgb_pixels <= 0.0031308).type(torch.FloatTensor).cuda()
		exponential_mask = (rgb_pixels > 0.0031308).type(torch.FloatTensor).cuda()
		srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (((rgb_pixels+0.000001) ** (1/2.4) * 1.055) - 0.055) * exponential_mask
	
		return torch.reshape(srgb_pixels, lab.shape)


def NLM_filtering(image, kernel_size=11, std=4.0,kernel_size_mean=3, sub_filter_size=32):
      
    lab_image = rgb_to_lab(image)
    channels = preprocess_lab(lab_image)
    output_nlm = torch.stack([
    nlm2d(channels[c], 
            kernel_size=11, # neighbourhood size 
            std=4.0, # the sigma 
            kernel_size_mean=3, # the kernel used to compute the average pixel intensity
            sub_filter_size=32 # how many neighbourhoods are computed per iteration
            )
            for c in range(channels.shape[0])
            ], dim=0)
    lab = deprocess_lab(output_nlm[0], output_nlm[1], output_nlm[2])
    output = lab_to_rgb(lab)
    return output
