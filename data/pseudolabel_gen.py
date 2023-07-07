import matplotlib.pyplot as plt
import copy
import skimage

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage.measure import label 
import scipy.ndimage.morphology as snm
from skimage import io
import argparse
import numpy as np
import glob

import SimpleITK as sitk
import os

to01 = lambda x: (x - x.min()) / (x.max() - x.min())

DATASET_CONFIG = {'SABS':{
                    'img_bname': f'./sabs_CT_normalized/image_*.nii.gz',
                    'out_dir': './sabs_CT_normalized',
                    'fg_thresh': 1e-4
                    },
                  'CHAOST2':{
                      'img_bname': f'../CHAOST2/chaos_MR_T2_normalized/image_*.nii.gz',
                      'out_dir': './CHAOST2/chaos_MR_T2_normalized',
                      'fg_thresh': 1e-4 + 50
                    }
                 }
            

DOMAIN = 'SABS'
img_bname = DATASET_CONFIG[DOMAIN]['img_bname']
imgs = glob.glob(img_bname)
out_dir = DATASET_CONFIG[DOMAIN]['out_dir']


imgs = sorted(imgs, key = lambda x: int(x.split('_')[-1].split('.nii.gz')[0]) )

MODE = 'MIDDLE' # minimum size of pesudolabels. 'MIDDLE' is the default setting

# wrapper for process 3d image in 2d
def superpix_vol(img, method = 'fezlen', **kwargs):
    """
    loop through the entire volume
    assuming image with axis z, x, y
    """
    if method =='fezlen':
        seg_func = skimage.segmentation.felzenszwalb
    else:
        raise NotImplementedError
        
    out_vol = np.zeros(img.shape)
    for ii in range(img.shape[0]):
        if MODE == 'MIDDLE':
            segs = seg_func(img[ii, ...], min_size = 400, sigma = 1)
        else:
            raise NotImplementedError
        out_vol[ii, ...] = segs
        
    return out_vol

# thresholding the intensity values to get a binary mask of the patient
def fg_mask2d(img_2d, thresh): # change this by your need
    mask_map = np.float32(img_2d > thresh)
    
    def getLargestCC(segmentation): # largest connected components
        labels = label(segmentation)
        assert( labels.max() != 0 ) # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        return largestCC
    if mask_map.max() < 0.999:
        return mask_map
    else:
        post_mask = getLargestCC(mask_map)
        fill_mask = snm.binary_fill_holes(post_mask)
    return fill_mask

# remove superpixels within the empty regions
def superpix_masking(raw_seg2d, mask2d):
    raw_seg2d = np.int32(raw_seg2d)
    lbvs = np.unique(raw_seg2d)
    max_lb = lbvs.max()
    raw_seg2d[raw_seg2d == 0] = max_lb + 1
    lbvs = list(lbvs)
    lbvs.append( max_lb )
    raw_seg2d = raw_seg2d * mask2d
    lb_new = 1
    out_seg2d = np.zeros(raw_seg2d.shape)
    for lbv in lbvs:
        if lbv == 0:
            continue
        else:
            out_seg2d[raw_seg2d == lbv] = lb_new
            lb_new += 1
    
    return out_seg2d
            
def superpix_wrapper(img, verbose = False, fg_thresh = 1e-4):
    raw_seg = superpix_vol(img)
    fg_mask_vol = np.zeros(raw_seg.shape)
    processed_seg_vol = np.zeros(raw_seg.shape)
    for ii in range(raw_seg.shape[0]):
        if verbose:
            print("doing {} slice".format(ii))
        _fgm = fg_mask2d(img[ii, ...], fg_thresh )
        _out_seg = superpix_masking(raw_seg[ii, ...], _fgm)
        fg_mask_vol[ii] = _fgm
        processed_seg_vol[ii] = _out_seg
    return fg_mask_vol, processed_seg_vol
        
# copy spacing and orientation info between sitk objects
def copy_info(src, dst):
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())
    # dst.CopyInfomation(src)
    return dst


def strip_(img, lb):
    img = np.int32(img)
    if isinstance(lb, float):
        lb = int(lb)
        return np.float32(img == lb) * float(lb)
    elif isinstance(lb, list):
        out = np.zeros(img.shape)
        for _lb in lb:
            out += np.float32(img == int(_lb)) * float(_lb)
            
        return out
    else:
        raise Exception

# Generate pseudolabels for every image and save them
for img_fid in imgs:
# img_fid = imgs[0]

    idx = os.path.basename(img_fid).split("_")[-1].split(".nii.gz")[0]
    im_obj = sitk.ReadImage(img_fid)

    out_fg, out_seg = superpix_wrapper(sitk.GetArrayFromImage(im_obj), fg_thresh = DATASET_CONFIG[DOMAIN]['fg_thresh'] )
    out_fg_o = sitk.GetImageFromArray(out_fg ) 
    out_seg_o = sitk.GetImageFromArray(out_seg )

    out_fg_o = copy_info(im_obj, out_fg_o)
    out_seg_o = copy_info(im_obj, out_seg_o)
    seg_fid = os.path.join(out_dir, f'superpix-{MODE}_{idx}.nii.gz')
    msk_fid = os.path.join(out_dir, f'fgmask_{idx}.nii.gz')
    sitk.WriteImage(out_fg_o, msk_fid)
    sitk.WriteImage(out_seg_o, seg_fid)
    print(f'image with id {idx} has finished')






