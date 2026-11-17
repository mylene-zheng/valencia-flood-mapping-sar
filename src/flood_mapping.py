"""Non-learning flood mapping with Sentinel-1 (Valencia DANA, 29 Oct 2024), on the SNAP product.

Input: valencia_flood_coregisteration.tif, made with SNAP from the Copernicus GRD products
(orbit, thermal noise removal, slice assembly, sigma0 calibration, coregistration, terrain correction, subset).
Band order: 1 VH post 01/11, 2 VV post 01/11, 3 VH pre 20/10, 4 VV pre 20/10 (linear sigma0).

Pipeline (all classical image processing, no training):
  1. Speckle reduction with the Lee filter (SAR speckle is multiplicative).
  2. Method A - single-date: Otsu threshold on post-event VV backscatter (dB).
  3. Method B - change detection: log-ratio  LR = 10 log10(VV_post / VV_pre),
     threshold chosen with Kittler-Illingworth (minimum error) on LR, combined
     with "post pixel is dark", and permanent water removed with JRC GSW.
  4. Spatial regularisation of method B with a Markov Random Field (Potts prior,
     ICM optimisation), as in the change-detection part of the lecture.
  5. Validation against an independent optical reference: Sentinel-2 MNDWI water
     masks of 31 Oct 2024 (and 26 Oct 2024 for 'new water') on cloud-free pixels.
"""
import json, pathlib
import numpy as np, rasterio
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = pathlib.Path(__file__).resolve().parent.parent
D, FIG = ROOT / "data", ROOT / "figures"
SNAP = ROOT / "valencia_flood_coregisteration.tif"
FIG.mkdir(exist_ok=True)
db = lambda x: 10 * np.log10(np.clip(x, 1e-5, None))

def load_snap():
    """Read the SNAP GeoTIFF. Pixels at 0 or NaN are set to NaN (none are expected in the final SNAP product)."""
    with rasterio.open(SNAP) as s:
        b = s.read().astype("float32")
    b[~np.isfinite(b) | (b <= 0)] = np.nan
    post, pre = {"VH": b[0], "VV": b[1]}, {"VH": b[2], "VV": b[3]}
    return pre, post

# ---------------------------------------------------------------- 1. speckle filter
def lee_filter(img, size=7, looks=4.4):
    """Classic Lee filter. Noise variation coefficient Cu = 1/sqrt(L) for an L-look intensity image
    (Sentinel-1 IW GRDH has an ENL of about 4.4)."""
    img = np.nan_to_num(img, nan=np.nanmedian(img))
    m = ndi.uniform_filter(img, size)
    m2 = ndi.uniform_filter(img ** 2, size)
    var = np.maximum(m2 - m ** 2, 0)
    cu2 = 1.0 / looks
    ci2 = var / np.maximum(m ** 2, 1e-12)
    w = np.clip(1 - cu2 / np.maximum(ci2, 1e-12), 0, 1)    # 0 on homogeneous areas, 1 on edges
    return m + w * (img - m)

def estimate_enl(img, win=15):
    """Equivalent number of looks: mean^2/var, taken on the most homogeneous windows."""
    m = ndi.uniform_filter(img, win); v = ndi.uniform_filter(img ** 2, win) - m ** 2
    enl = m ** 2 / np.maximum(v, 1e-12)
    return float(np.nanpercentile(enl[np.isfinite(enl)], 90))

# ---------------------------------------------------------------- 2/3. thresholds
def kittler_illingworth(x, bins=256):
    """Minimum-error thresholding (Kittler & Illingworth 1986): fits two Gaussians to the histogram."""
    h, e = np.histogram(x, bins=bins); c = (e[:-1] + e[1:]) / 2; h = h / h.sum()
    best, t_best = np.inf, c[len(c) // 2]
    for i in range(2, bins - 2):
        p1, p2 = h[:i].sum(), h[i:].sum()
        if p1 < 1e-4 or p2 < 1e-4: continue
        m1 = (h[:i] * c[:i]).sum() / p1; m2 = (h[i:] * c[i:]).sum() / p2
        s1 = np.sqrt((h[:i] * (c[:i] - m1) ** 2).sum() / p1); s2 = np.sqrt((h[i:] * (c[i:] - m2) ** 2).sum() / p2)
        if s1 < 1e-6 or s2 < 1e-6: continue
        J = 1 + 2 * (p1 * np.log(s1) + p2 * np.log(s2)) - 2 * (p1 * np.log(p1) + p2 * np.log(p2))
        if J < best: best, t_best = J, c[i]
    return float(t_best)

# ---------------------------------------------------------------- 4. MRF / ICM
def icm_potts(feature, init, beta=1.5, iters=8):
    """Two-class MRF: data term = Gaussian negative log-likelihood of each class (parameters re-estimated
    every iteration), context term = Potts model on the 8-neighbourhood. Optimised with ICM (Besag 1986)."""
    lab = init.astype(np.uint8)
    k = np.ones((3, 3)); k[1, 1] = 0
    for _ in range(iters):
        U = []
        for c in (0, 1):
            x = feature[lab == c]; mu, sd = x.mean(), x.std() + 1e-6
            data = 0.5 * ((feature - mu) / sd) ** 2 + np.log(sd) - np.log((lab == c).mean())
            same = ndi.convolve((lab == c).astype(float), k, mode="nearest")
            U.append(data + beta * (8 - same))
        new = (U[1] < U[0]).astype(np.uint8)
        changed = (new != lab).mean(); lab = new
        if changed < 1e-4: break
    return lab.astype(bool)

def scores(pred, ref, valid):
    p, r = pred[valid], ref[valid]
    tp = np.sum(p & r); fp = np.sum(p & ~r); fn = np.sum(~p & r); tn = np.sum(~p & ~r)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(precision=round(prec, 3), recall=round(rec, 3),
                F1=round(2 * prec * rec / max(prec + rec, 1e-9), 3), IoU=round(tp / max(tp + fp + fn, 1), 3),
                overall_accuracy=round((tp + tn) / p.size, 3), kappa=round(_kappa(tp, fp, fn, tn), 3))

def _kappa(tp, fp, fn, tn):
    n = tp + fp + fn + tn; po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return (po - pe) / (1 - pe)

def optical_references(permanent):
    """Two optical references from Sentinel-2 L2A (processing baseline >= 04.00:
    subtract the 1000 offset added by ESA to the reflectances).
    R1 'post-event water': MNDWI water on 31/10 outside permanent water, on cloud-free pixels.
    R2 'new water'       : R1 minus pixels already water on the 26/10 pre-event image."""
    s2 = np.load(D / "s2_2024-10-31_post.npz"); s0 = np.load(D / "s2_2024-10-26_pre.npz")
    mndwi = lambda d: ((d["B03"] - 1000.0) - (d["B11"] - 1000.0)) / np.maximum(d["B03"] + d["B11"] - 2000.0, 1)
    clear = lambda d: np.isin(d["SCL"], [2, 4, 5, 6, 7, 11])   # flood water is turbid: SCL often calls it soil,
    m1, m0 = mndwi(s2), mndwi(s0)                               # so water is taken from MNDWI, not from SCL
    tW = max(float(threshold_otsu(m1[clear(s2) & np.isfinite(m1)])), 0.0)
    w1, w0 = (m1 > tW) & clear(s2), (m0 > tW) & clear(s0)
    R1 = (w1 & ~permanent, clear(s2) & ~permanent)
    R2 = (w1 & ~w0 & ~permanent, clear(s2) & clear(s0) & ~permanent)
    return R1, R2, tW, m1, clear(s2), s2


def map_floods(pre, post, permanent):
    valid = np.isfinite(pre["VV"]) & np.isfinite(post["VV"]) & (pre["VV"] > 0) & (post["VV"] > 0)
    vv_pre_db, vv_post_db = db(lee_filter(pre["VV"])), db(lee_filter(post["VV"]))
    lr = vv_post_db - vv_pre_db                                    # log-ratio in dB (speckle is multiplicative)
    # Method A: single-date Otsu on post VV
    tA = float(threshold_otsu(vv_post_db[valid]))
    A = (vv_post_db < tA) & valid & ~permanent
    # Method B: change detection, KI threshold on the decreasing part of the log-ratio + dark after
    tB = kittler_illingworth(lr[valid & (lr < 0)])
    B = (lr < tB) & (vv_post_db < tA) & valid & ~permanent
    # Method C: B regularised by an MRF (Potts prior, ICM)
    C = icm_potts(np.where(valid, lr, 0), B) & (vv_post_db < tA + 2) & valid & ~permanent
    # Ablation: B without speckle filter
    raw_lr = db(np.nan_to_num(post["VV"], nan=1)) - db(np.nan_to_num(pre["VV"], nan=1))
    B_raw = (raw_lr < tB) & (db(np.nan_to_num(post["VV"], nan=1)) < tA) & valid & ~permanent
    return dict(A=A, B=B, C=C, B_raw=B_raw), dict(otsu_post_VV_dB=round(tA, 2), KI_log_ratio_dB=round(tB, 2)), \
        lr, vv_post_db, valid


def main():
    pre, post = load_snap()
    gsw = np.load(D / "jrc_gsw_occurrence.npz")["occurrence"]
    permanent = np.nan_to_num(gsw) >= 50                           # water > 50 % of the time since 1984 (JRC GSW)
    (R1, V1), (R2, V2), tW, mndwi, clear, s2 = optical_references(permanent)
    maps, thr, lr, vv_post_db, valid = map_floods(pre, post, permanent)
    res = {"input": SNAP.name,
           "ENL_estimated_VV": {"pre": round(estimate_enl(np.nan_to_num(pre["VV"], nan=0.1)), 2),
                                "post": round(estimate_enl(np.nan_to_num(post["VV"], nan=0.1)), 2)},
           "otsu_MNDWI_threshold": round(tW, 3),
           "reference_area_km2": {"R1_post_water": round(R1.sum() * 1e-4, 2), "R2_new_water": round(R2.sum() * 1e-4, 2)},
           "thresholds": thr, "methods": {}}
    names = {"A": "single-date Otsu", "B": "log-ratio + KI", "C": "log-ratio + KI + MRF", "B_raw": "log-ratio + KI, no Lee filter"}
    for k, m in maps.items():
        res["methods"][f"{k}: {names[k]}"] = {"area_km2": round(m.sum() * 1e-4, 2),
                                              "vs_R1_post_water": scores(m, R1, V1), "vs_R2_new_water": scores(m, R2, V2)}
    # same methods on the VH (cross-polarised) channel, for comparison with VV
    maps_vh, thr_vh, *_ = map_floods({"VV": pre["VH"]}, {"VV": post["VH"]}, permanent)
    res["thresholds_VH"], res["methods_VH"] = thr_vh, {}
    for k, m in maps_vh.items():
        if k == "B_raw": continue
        res["methods_VH"][f"{k}: {names[k]} (VH)"] = {"area_km2": round(m.sum() * 1e-4, 2),
                                                      "vs_R1_post_water": scores(m, R1, V1), "vs_R2_new_water": scores(m, R2, V2)}
    json.dump(res, open(ROOT / "results.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))

    flood_C, flood_B, tA, tB = maps["C"], maps["B"], thr["otsu_post_VV_dB"], thr["KI_log_ratio_dB"]
    np.savez_compressed(D / "flood_maps.npz", **maps, R1=R1, V1=V1, R2=R2, V2=V2, permanent=permanent)

    # ------------------------------------------------------------ figures
    rgb = lambda d: np.clip((np.dstack([d["B11"], d["B08"], d["B04"]]) - 1000) / 3500, 0, 1)  # water = dark blue
    ext = dict(interpolation="nearest")
    f, ax = plt.subplots(2, 3, figsize=(15, 16))
    ax[0, 0].imshow(db(pre["VV"]), cmap="gray", vmin=-22, vmax=3, **ext); ax[0, 0].set_title("S1 VV 20/10/2024 (pre, SNAP sigma0)")
    ax[0, 1].imshow(vv_post_db, cmap="gray", vmin=-22, vmax=3, **ext); ax[0, 1].set_title("S1 VV 01/11/2024 (post, Lee 7x7)")
    im = ax[0, 2].imshow(lr, cmap="RdBu", vmin=-10, vmax=10, **ext); ax[0, 2].set_title("Log-ratio post/pre (dB)")
    plt.colorbar(im, ax=ax[0, 2], fraction=0.04)
    ax[1, 0].imshow(rgb(s2), **ext); ax[1, 0].set_title("S2 31/10/2024 (SWIR/NIR/R), clouds in the west")
    over = np.zeros(lr.shape, np.uint8); over[permanent] = 1; over[flood_C] = 2
    ax[1, 1].imshow(vv_post_db, cmap="gray", vmin=-22, vmax=3, **ext)
    ax[1, 1].imshow(np.ma.masked_equal(over, 0), cmap=ListedColormap(["#1f4e9c", "#e8412c"]), vmin=1, vmax=2, alpha=.75, **ext)
    ax[1, 1].set_title("Flood (red) / permanent water (blue)\nlog-ratio + KI + MRF")
    cmp_ = np.zeros(lr.shape, np.uint8)
    cmp_[V1 & flood_C & R1] = 1; cmp_[V1 & flood_C & ~R1] = 2; cmp_[V1 & ~flood_C & R1] = 3; cmp_[~V1] = 4
    ax[1, 2].imshow(cmp_, cmap=ListedColormap(["white", "#2a9d4b", "#e8412c", "#f2a33a", "#cccccc"]), vmin=0, vmax=4, **ext)
    ax[1, 2].set_title("vs S2 water 31/10: TP green, FP red,\nFN orange, cloud/permanent water grey")
    for a in ax.flat: a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(FIG / "overview.png", dpi=100); plt.close()

    f, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].hist(vv_post_db[valid], 200, range=(-30, 10), color="#555"); ax[0].axvline(tA, color="r")
    ax[0].set_title(f"Post VV (dB), Otsu = {tA:.1f} dB")
    ax[1].hist(lr[valid], 200, range=(-15, 10), color="#555", log=True); ax[1].axvline(tB, color="r")
    ax[1].set_title(f"Log-ratio (dB), Kittler-Illingworth = {tB:.1f} dB")
    ax[2].hist(mndwi[clear & np.isfinite(mndwi)], 200, range=(-1, 1), color="#555"); ax[2].axvline(tW, color="r")
    ax[2].set_title(f"S2 MNDWI 31/10 on clear pixels, Otsu = {tW:.2f}")
    plt.tight_layout(); plt.savefig(FIG / "histograms.png", dpi=110); plt.close()

    sl = np.s_[1300:1800, 1000:1500]                              # rice fields north-west of the Albufera
    f, ax = plt.subplots(1, 4, figsize=(18, 5))
    ax[0].imshow(db(post["VV"])[sl], cmap="gray", vmin=-22, vmax=3); ax[0].set_title("raw post VV (speckle)")
    ax[1].imshow(vv_post_db[sl], cmap="gray", vmin=-22, vmax=3); ax[1].set_title("Lee filtered")
    ax[2].imshow(flood_B[sl], cmap="gray"); ax[2].set_title("log-ratio + KI")
    ax[3].imshow(flood_C[sl], cmap="gray"); ax[3].set_title("+ MRF (ICM, Potts)")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(FIG / "zoom_filtering_mrf.png", dpi=110); plt.close()


if __name__ == "__main__":
    main()
