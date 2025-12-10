import numpy as np
from scipy.signal import convolve2d
import interpolation as intrp
def kernel_1d(K, method):
    x = np.linspace(-(K//2), K//2, K)

    if method == "NN":
        # klasyczny BOX FILTER
        k = np.ones(K, dtype=np.float64)

    elif method == "linear":
        k = 1 - np.abs(x) / (K/2)
        k[k < 0] = 0

    elif method == "cubic":
        a = -0.5
        absx = np.abs(x) / (K/2)
        k = np.zeros_like(absx)

        m1 = absx <= 1
        m2 = (absx > 1) & (absx < 2)

        k[m1] = (a+2)*absx[m1]**3 - (a+3)*absx[m1]**2 + 1
        k[m2] = a*absx[m2]**3 - 5*a*absx[m2]**2 + 8*a*absx[m2] - 4*a

    else:
        raise ValueError("Unknown method")

    k = k.astype(np.float64)

    s = k.sum()
    if s == 0:
        raise RuntimeError(f"Kernel sum is zero! method={method}, K={K}")

    k /= s
    return k

def smooth_image(img, K, method):
    img = img.astype(np.float64)

    k = kernel_1d(K, method)
    kernel2d = np.outer(k, k)

    if img.ndim == 3:
        out = np.zeros(img.shape, dtype=np.float64)
        for c in range(3):
            out[..., c] = convolve2d(
                img[..., c], kernel2d,
                mode="same", boundary="symm"
            )
    else:
        out = convolve2d(img, kernel2d, mode="same", boundary="symm")

    return np.clip(out, 0.0, 1.0)


def resize(img, newH, newW, method):
    img = img.astype(np.float64)

    H, W = img.shape[:2]

    xs, ys = np.meshgrid(
        np.linspace(0, W-1, newW),
        np.linspace(0, H-1, newH)
    )

    out = intrp.interp2d(img, xs, ys, method)
    return np.clip(out, 0.0, 1.0)
