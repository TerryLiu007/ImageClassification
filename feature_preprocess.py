import cv2
from mahotas.features import surf, haralick
from skimage.feature import hog
from skimage.feature import local_binary_pattern as lbp
import numpy as np


def LBP(numPoints, radius, image, eps=1e-7):
    m_lbp = lbp(image, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(m_lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    
    assert (hist.shape == (26,)), "LBP shape expected 26, got {}".format(hist.shape)
    return hist.ravel()


def H_Tex(image, eps=1e-7):
    m_haralick = haralick(image)
    assert m_haralick.shape == (4,13)
    # normalize wrt each row
    row_sum = m_haralick.sum(axis=1)
    m_haralick = m_haralick/(row_sum[:,np.newaxis] + eps)
    
    assert (m_haralick.shape == (4,13)), "haralick shape expected (4,13), got {}".format(m_haralick.shape)
    return m_haralick.ravel()


def SURF(image):
    m_surf = surf.surf(image, max_points=512, descriptor_only=True)
    m_surf = m_surf[:10,:]
    m_surf = np.append(m_surf, np.zeros((10-m_surf.shape[0],64)))
    
    assert (m_surf.shape == (640,)), "SURF shape expected (640,), got {}".format(m_surf.shape)
    return m_surf.ravel()


def HOG(image):
    img = cv2.resize(image, (55,55))
    m_hog = hog(img, block_norm='L1')
    
    return m_hog


def feature_vector(image):
    x = []
    x.extend(LBP(24,8,image))
    x.extend(H_Tex(image))
    x.extend(SURF(image))
    x.extend(HOG(image))
    
    x = np.asarray(x)
    assert x.shape == (2014,)
    return x

