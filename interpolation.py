import numpy as np

def NN(tx, f0, f1):
    f0 = f0.astype(float)
    f1 = f1.astype(float)
    tx = tx[...,None]
    return np.where(tx < 0.5, f0, f1)

def linear(tx, f0, f1):
    f0 = f0.astype(float)
    f1 = f1.astype(float)
    tx = tx.astype(float)
    tx = tx[...,None]
    return f0 + tx * (f1 - f0)


# ============================================================
# FAST CUBIC CATMULL-ROM
# ============================================================

def cubic(p0, p1, p2, p3, t):
    # konwersja na float (konieczne!)
    p0 = p0.astype(float)
    p1 = p1.astype(float)
    p2 = p2.astype(float)
    p3 = p3.astype(float)
    t  = t.astype(float)
    t  = t[..., None]
    t2 = t * t
    t3 = t2 * t

    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t +
        (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
        (-p0 + 3*p1 - 3*p2 + p3) * t3
    )




# ============================================================
# FAST 2D INTERPOLATION
# ============================================================

def interp2d(img, xs, ys, method):
    H, W = img.shape[:2]

    # floor + clip w jednej operacji
    x0 = xs.astype(np.int32)
    y0 = ys.astype(np.int32)

    np.clip(x0, 0, W - 2, out=x0)
    np.clip(y0, 0, H - 2, out=y0)

    x1 = x0 + 1
    y1 = y0 + 1

    tx = xs - x0
    ty = ys - y0

    # szybkie pobieranie pikseli
    p00 = img[y0, x0]
    p01 = img[y0, x1]
    p10 = img[y1, x0]
    p11 = img[y1, x1]

    # ------------ NN ------------
    if method == "NN":
        fx0 = NN(tx, p00, p01)
        fx1 = NN(tx, p10, p11)
        return NN(ty, fx0, fx1)

    # ------------ LINEAR ------------
    if method == "linear":
        fx0 = linear(tx, p00, p01)
        fx1 = linear(tx, p10, p11)
        return linear(ty, fx0, fx1)

    # ------------ CUBIC ------------
    if method == "cubic":
        # fallback dla ma�ych obraz�w
        if W < 4 or H < 4:
            return interp2d(img.astype(float), xs, ys, "linear")

        xi = x0
        yi = y0

        xf = xs - xi   # fractional x
        yf = ys - yi   # fractional y

        # neighbour indices (correct order)
        xm1 = np.clip(xi - 1, 0, W-1)
        x0c = xi
        x1c = np.clip(xi + 1, 0, W-1)
        x2c = np.clip(xi + 2, 0, W-1)

        ym1 = np.clip(yi - 1, 0, H-1)
        y0c = yi
        y1c = np.clip(yi + 1, 0, H-1)
        y2c = np.clip(yi + 2, 0, H-1)

        # ROW interpolation (pion)
        R0 = cubic(img[ym1, xm1], img[ym1, x0c], img[ym1, x1c], img[ym1, x2c], xf)
        R1 = cubic(img[y0c, xm1], img[y0c, x0c], img[y0c, x1c], img[y0c, x2c], xf)
        R2 = cubic(img[y1c, xm1], img[y1c, x0c], img[y1c, x1c], img[y1c, x2c], xf)
        R3 = cubic(img[y2c, xm1], img[y2c, x0c], img[y2c, x1c], img[y2c, x2c], xf)

        # COLUMN interpolation (poziom)
        return cubic(R0, R1, R2, R3, yf)


    raise ValueError("Unknown method")


def sample_function_1d(f, xmin, xmax, K):
    xs = np.linspace(xmin, xmax, K)
    ys = f(xs)
    return xs, ys