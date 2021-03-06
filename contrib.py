import numpy as np
import tensorflow as tf
import pydensecrf.densecrf as dcrf

def dense_crf(probs, class_num, img=None, n_iters=10, 
              sxy_gaussian=(1, 1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.
    
    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      
    Returns:
      Refined predictions after MAP inference.
    """
    n, h, w, _ = probs.shape
    
    preds = np.zeros_like(probs)
    
    for i in range(n):
        probs_i = probs[i].transpose(2, 0, 1).copy(order='C') + 1e-8 # Need a contiguous array.

        d = dcrf.DenseCRF2D(w, h, class_num) # Define DenseCRF model.
        U = -np.log(probs_i) # Unary potential.
        U = U.reshape((class_num, -1)) # Needs to be flat.
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                              kernel=kernel_gaussian, normalization=normalisation_gaussian)
        if img is not None:
            assert(img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
            img = img.astype(np.uint8)
            d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                                   kernel=kernel_bilateral, normalization=normalisation_bilateral,
                                   srgb=srgb_bilateral, rgbim=img[i])
        Q = d.inference(n_iters)
        preds[i, ...] = np.array(Q, dtype=np.float32).reshape((class_num, h, w)).transpose(1, 2, 0)
    
    return preds