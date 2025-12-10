# -*- coding: utf-8 -*-

import numpy as np
import interpolation as intrp
import matplotlib.pyplot as plt
import funcs as fnc
import os
import csv

# ==============================
# METRYKI
# ==============================

def MSE(original, reconstructed):
    return np.mean((original - reconstructed) ** 2)

def MAE(original, reconstructed):
    return np.mean(np.abs(original - reconstructed))

def add_poisson_noise(img, η):
    noisy = np.random.poisson(img / η) * η
    return np.clip(noisy, 0.0, 1.0)


# ==============================
# KATALOGI WYJŚCIOWE
# ==============================

os.makedirs("img/rozgrzewka", exist_ok=True)
os.makedirs("img/zestawienia_bez_szumu", exist_ok=True)
os.makedirs("img/zestawienia_z_szumem", exist_ok=True)
os.makedirs("csv", exist_ok=True)

csv_bez_szumu = open("csv/wyniki_bez_szumu.csv", "w", newline="", encoding="utf-8")
csv_szum = open("csv/wyniki_z_szumem.csv", "w", newline="", encoding="utf-8")

writer_bez = csv.writer(csv_bez_szumu)
writer_szum = csv.writer(csv_szum)

writer_bez.writerow(["method", "K", "p", "MSE", "MAE"])
writer_szum.writerow(["method", "eta", "K", "p", "MSE", "MAE"])


# ==============================
# 1. ROZGRZEWKA
# ==============================

img = plt.imread("img/karnegia-olbrzymia.jpg").astype(np.float64) / 255.0

H, W = img.shape[:2]
scale = 4.0

xs, ys = np.meshgrid(
    np.linspace(0, W - 1, int(W * scale)),
    np.linspace(0, H - 1, int(H * scale)),
)

methods = ["NN", "linear", "cubic"]

plt.figure(figsize=(10, 5))
for i, m in enumerate(methods):
    out = intrp.interp2d(img, xs, ys, m)
    plt.subplot(1, 3, i + 1)
    plt.title(m)
    plt.imshow(np.clip(out, 0, 1))
    plt.axis("off")
    plt.savefig(f"img/rozgrzewka/{m}.png")
plt.show()


# ==============================
# 2. BEZ SZUMU
# ==============================

img = plt.imread("img/karnegia-olbrzymia.jpg").astype(np.float64) / 255.0
img = img[:1024, :1024]

Ks = [3, 6, 12]
ps = [7, 8]   # 128 i 256

print("\n===== WERSJE BEZ SZUMU =====")

for method in methods:

    fig, axes = plt.subplots(len(Ks), len(ps), figsize=(10, 8))
    fig.suptitle(f"BEZ SZUMU — metoda: {method}", fontsize=14)

    for i, K in enumerate(Ks):

        img_smooth = fnc.smooth_image(img, K, method)

        for j, p in enumerate(ps):
            size = 2**p

            img_small = fnc.resize(img_smooth, size, size, method)
            img_back  = fnc.resize(img_small, 1024, 1024, method)

            mse = MSE(img, img_back)
            mae = MAE(img, img_back)

            print(f"{method}, K={K}, p={p}:  MSE={mse:.6f}, MAE={mae:.6f}")

            writer_bez.writerow([method, K, p, mse, mae])

            axes[i, j].imshow(img_back)
            axes[i, j].set_title(f"K={K}, p={p}\nMSE={mse:.4f}\nMAE={mae:.4f}")
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(f"img/zestawienia_bez_szumu/{method}.png")
    plt.show()


# ==============================
# 3. SZUM POISSONA
# ==============================

Η = [1, 4, 16, 64, 256]

print("\n===== WERSJE Z SZUMEM =====")

for method in methods:
    for η in Η:

        fig, axes = plt.subplots(len(Ks), len(ps), figsize=(10, 8))
        fig.suptitle(f"Z SZUMEM — metoda: {method}, η={η}", fontsize=14)

        img_noisy = add_poisson_noise(img, η)

        for i, K in enumerate(Ks):

            img_smooth = fnc.smooth_image(img_noisy, K, method)

            for j, p in enumerate(ps):
                size = 2**p

                img_small = fnc.resize(img_smooth, size, size, method)
                img_back  = fnc.resize(img_small, 1024, 1024, method)

                mse = MSE(img, img_back)
                mae = MAE(img, img_back)

                print(
                    f"{method}, η={η}, K={K}, p={p}: "
                    f"MSE={mse:.6f}, MAE={mae:.6f} | ZASZUMIONE"
                )

                writer_szum.writerow([method, η, K, p, mse, mae])

                axes[i, j].imshow(img_back)
                axes[i, j].set_title(f"K={K}, p={p}\nMSE={mse:.4f}\nMAE={mae:.4f}")
                axes[i, j].axis("off")

        plt.tight_layout()
        plt.savefig(f"img/zestawienia_z_szumem/{method}_eta_{η}.png")
        plt.show()


# ==============================
# ZAMKNIĘCIE PLIKÓW CSV
# ==============================

csv_bez_szumu.close()
csv_szum.close()

print("\n✅ Wyniki zapisane do:")
print("csv/wyniki_bez_szumu.csv")
print("csv/wyniki_z_szumem.csv")
