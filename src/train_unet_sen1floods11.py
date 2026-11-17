"""Optional part: vanilla deep-learning baseline on Sen1Floods11 (hand-labelled split).

This is the script used to train the U-Net whose weights are in data/unet_sen1floods11.pt
(40 epochs, about 8 minutes on an Apple MPS GPU). It is given to show the training code; the
case study does not retrain it. Data: run src/download_sen1floods11.py first (about 680 MB).

Compares on the official test split:
  - non-learning: Otsu threshold on VV (dB), computed per chip, and one global threshold learnt on train
  - learning    : small vanilla U-Net (VV, VH in dB), trained from scratch, BCE loss
The application to the Valencia SNAP image is in src/unet_valencia.py.
Note: running this script overwrites data/unet_sen1floods11.pt with a newly trained network.
"""
import json, pathlib, time
import numpy as np, rasterio, torch, torch.nn as nn, torch.nn.functional as F
from skimage.filters import threshold_otsu

ROOT = pathlib.Path(__file__).resolve().parent.parent
S = ROOT / "data" / "sen1floods11"
torch.manual_seed(0); np.random.seed(0)
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = np.array([-10.4, -17.1]), np.array([4.1, 4.4])      # dB statistics (VV, VH), normalisation only

def load(split):
    X, Y = [], []
    for line in open(S / f"flood_{split}_data.csv").read().split():
        a, b = line.split(",")
        x = rasterio.open(S / "S1Hand" / a).read().astype("float32")        # (2, 512, 512) VV, VH in dB
        y = rasterio.open(S / "LabelHand" / b).read(1).astype("int64")       # -1 nodata, 0 dry, 1 water
        bad = ~np.isfinite(x).all(0); y[bad] = -1
        x = np.nan_to_num(x, nan=0.0); x = np.clip(x, -50, 5)
        X.append(x); Y.append(y)
    return np.stack(X), np.stack(Y)

def norm(x): return (x - MEAN[None, :, None, None]) / STD[None, :, None, None]

class UNet(nn.Module):
    def __init__(s, c=16):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
        s.e1, s.e2, s.e3, s.e4 = blk(2, c), blk(c, 2 * c), blk(2 * c, 4 * c), blk(4 * c, 8 * c)
        s.u3, s.d3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, 2), blk(8 * c, 4 * c)
        s.u2, s.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), blk(4 * c, 2 * c)
        s.u1, s.d1 = nn.ConvTranspose2d(2 * c, c, 2, 2), blk(2 * c, c)
        s.out = nn.Conv2d(c, 1, 1)
    def forward(s, x):
        e1 = s.e1(x); e2 = s.e2(F.max_pool2d(e1, 2)); e3 = s.e3(F.max_pool2d(e2, 2)); e4 = s.e4(F.max_pool2d(e3, 2))
        d3 = s.d3(torch.cat([s.u3(e4), e3], 1)); d2 = s.d2(torch.cat([s.u2(d3), e2], 1)); d1 = s.d1(torch.cat([s.u1(d2), e1], 1))
        return s.out(d1)[:, 0]

def iou_stats(pred, y):
    v = y >= 0; p, t = pred[v].astype(bool), y[v] == 1
    return np.array([np.sum(p & t), np.sum(p & ~t), np.sum(~p & t), np.sum(~p & ~t)], dtype=float)

def summarize(st):
    tp, fp, fn, tn = st; prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(IoU=round(tp / (tp + fp + fn), 3), F1=round(2 * prec * rec / (prec + rec + 1e-9), 3),
                precision=round(prec, 3), recall=round(rec, 3), overall_accuracy=round((tp + tn) / st.sum(), 3))

def predict(model, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(norm(x)).float().to(DEV))).cpu().numpy()

def main(epochs=40):
    Xtr, Ytr = load("train"); Xva, Yva = load("valid"); Xte, Yte = load("test")
    print("chips train/valid/test:", len(Xtr), len(Xva), len(Xte), "water fraction train:",
          round(float((Ytr == 1).sum() / (Ytr >= 0).sum()), 3))
    res = {}
    # ---- non-learning baselines on VV
    st = np.zeros(4)
    for x, y in zip(Xte, Yte):
        vv = x[0][y >= 0]
        t = threshold_otsu(vv) if vv.size and vv.max() > vv.min() else -15
        st += iou_stats(x[0] < t, y)
    res["Otsu per chip (VV)"] = summarize(st)
    t_glob = threshold_otsu(Xtr[:, 0][Ytr >= 0])
    st = sum(iou_stats(x[0] < t_glob, y) for x, y in zip(Xte, Yte))
    res[f"global Otsu from train (VV < {t_glob:.1f} dB)"] = summarize(st)

    # ---- vanilla U-Net
    model = UNet().to(DEV); opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pw = torch.tensor(float((Ytr == 0).sum() / max((Ytr == 1).sum(), 1)) ** 0.5, device=DEV)  # mild class balancing
    Xtr_n = norm(Xtr); best, best_state, t0 = -1, None, time.time()
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(len(Xtr))
        for i in range(0, len(perm), 8):
            idx = perm[i:i + 8]; r, c = np.random.randint(0, 257, 2)
            xb = torch.from_numpy(Xtr_n[idx, :, r:r + 256, c:c + 256]).float()
            yb = torch.from_numpy(Ytr[idx, r:r + 256, c:c + 256])
            if np.random.rand() < .5: xb, yb = xb.flip(-1), yb.flip(-1)
            if np.random.rand() < .5: xb, yb = xb.flip(-2), yb.flip(-2)
            xb, yb = xb.to(DEV), yb.to(DEV); m = yb >= 0
            loss = F.binary_cross_entropy_with_logits(model(xb)[m], yb[m].float(), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        pv = np.concatenate([predict(model, Xva[i:i + 8]) for i in range(0, len(Xva), 8)])
        v_iou = summarize(sum(iou_stats(p > .5, y) for p, y in zip(pv, Yva)))["IoU"]
        if v_iou > best: best, best_state = v_iou, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep + 1:02d} loss {loss.item():.3f} val IoU {v_iou:.3f} ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state); torch.save(best_state, ROOT / "data" / "unet_sen1floods11.pt")
    pt = np.concatenate([predict(model, Xte[i:i + 8]) for i in range(0, len(Xte), 8)])
    res["vanilla U-Net (VV+VH, from scratch)"] = summarize(sum(iou_stats(p > .5, y) for p, y in zip(pt, Yte)))
    res["U-Net best validation IoU"] = best

    json.dump(res, open(ROOT / "results_sen1floods11.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    k = np.argsort([-(y == 1).mean() for y in Yte])[[2, 10, 25]]
    f, ax = plt.subplots(3, 4, figsize=(14, 10.5))
    for r, i in enumerate(k):
        vv = Xte[i, 0]; y = Yte[i]
        t = threshold_otsu(vv[y >= 0])
        for c, (img, ttl) in enumerate([(vv, "VV (dB)"), (np.where(y < 0, .5, y), "hand label"),
                                         (vv < t, f"Otsu (t={t:.1f} dB)"), (pt[i] > .5, "U-Net")]):
            ax[r, c].imshow(img, cmap="gray" if c else "gray", vmin=(-25 if c == 0 else 0), vmax=(0 if c == 0 else 1))
            ax[r, c].set_title(ttl); ax[r, c].axis("off")
    plt.tight_layout(); plt.savefig(ROOT / "figures" / "sen1floods11_examples.png", dpi=90); plt.close()

if __name__ == "__main__":
    main()
