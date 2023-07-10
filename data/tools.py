import numpy as np
import cv2 as cv
from PIL import Image
# import pydensecrf.densecrf as dcrf
from skimage.filters import threshold_otsu


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _memoryview_safe(x):
    """Make array safe to run in a Cython memoryview-based kernel. These
    kernels typically break down with the error ``ValueError: buffer source
    array is read-only`` when running in dask distributed.
    from: https://github.com/dask/distributed/issues/1978
    """
    if not x.flags.writeable:
        if not x.flags.owndata:
            x = x.copy(order='C')
        x.setflags(write=True)
    return x


# Fed the prediction results to a (CRF) for
# classifying image pixels as shadows/non-shadows.
def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=3)

    img = _memoryview_safe(img)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')


def post_processing(image, prediction, processing_method='CRF'):
    if processing_method == 'CRF':
        result = crf_refine(np.asarray(image).astype(np.uint8), np.asarray(prediction).astype(np.uint8))

    elif processing_method == 'THR':
        threshold = threshold_otsu(np.asarray(prediction))
        _, result = cv.threshold(np.asarray(prediction), threshold, 255, cv.THRESH_BINARY)

    # cv.imshow('Original', np.asarray(image))
    # cv.imshow('before', np.asarray(prediction))
    # cv.imshow('after', np.asarray(result))
    # cv.waitKey(0)

    return result