"""Optional part: apply the vanilla U-Net trained on Sen1Floods11 (hand-labelled split) to the SNAP
post-event image, and score it against the Sentinel-2 reference R1, like method A.

The network was trained already (src/train_unet_sen1floods11.py, 40 epochs, test IoU 0.65);
its weights are stored in data/unet_sen1floods11.pt. It is not retrained here.
Sen1Floods11 chips are Earth Engine sigma0 GRD (orbit, noise removal, calibration, terrain correction),
which is the same kind of product as the SNAP output, so the input given to the network matches its training data."""
import json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flood_mapping import load_snap, scores, ROOT, D, FIG

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = np.array([-10.4, -17.1]), np.array([4.1, 4.4])      # dB statistics (VV, VH) used in training

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

def main():
    model = UNet().to(DEV); model.load_state_dict(torch.load(D / "unet_sen1floods11.pt", map_location=DEV)); model.eval()
    _, post = load_snap(); maps = np.load(D / "flood_maps.npz")
    x = np.stack([10 * np.log10(np.clip(np.nan_to_num(post[b], nan=1e-5), 1e-5, None)) for b in ("VV", "VH")])
    x = (np.clip(x, -50, 5) - MEAN[:, None, None]) / STD[:, None, None]
    H, W = x.shape[1:]; Hp, Wp = (H + 7) // 8 * 8, (W + 7) // 8 * 8
    xp = np.pad(x, ((0, 0), (0, Hp - H), (0, Wp - W)), mode="reflect")[None]
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(xp).float().to(DEV))).cpu().numpy()[0, :H, :W]
    water = (prob > .5) & ~maps["permanent"]
    R1, V1 = maps["R1"], maps["V1"]
    res = {"Valencia (SNAP): U-Net post-event water vs S2 R1": scores(water, R1, V1),
           "Valencia (SNAP): method A (Otsu) vs S2 R1, for comparison": scores(maps["A"], R1, V1),
           "U-Net water area (km2)": round(water.sum() * 1e-4, 2)}
    json.dump(res, open(ROOT / "results_dl.json", "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))

    f, ax = plt.subplots(1, 2, figsize=(11, 8))
    ax[0].imshow(maps["A"], cmap="Blues"); ax[0].set_title("Valencia: single-date Otsu (method A)")
    ax[1].imshow(water, cmap="Blues"); ax[1].set_title("Valencia: U-Net trained on Sen1Floods11")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig(FIG / "valencia_unet_vs_otsu.png", dpi=90); plt.close()

if __name__ == "__main__":
    main()
