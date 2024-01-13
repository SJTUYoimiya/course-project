import numpy as np


def MSR(img, img_denoised):
    return np.sum((img - img_denoised)**2) / img.size


def PSNR(img, img_denoised):
    return 10 * np.log10(1 / MSR(img, img_denoised))
