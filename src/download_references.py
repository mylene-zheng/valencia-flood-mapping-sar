"""Download the validation layers directly onto the grid of the SNAP product (no account needed):
  - Sentinel-2 L2A of 26/10/2024 (pre) and 31/10/2024 (post), bands B03, B04, B08, B11 and SCL,
  - JRC Global Surface Water occurrence (permanent water mask).
They come from Microsoft Planetary Computer and are resampled once, onto the SNAP grid (UTM 31N, 10 m)."""
import pathlib
import numpy as np, rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.windows import from_bounds
import pystac_client, planetary_computer

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNAP = ROOT / "valencia_flood_coregisteration.tif"
OUT = ROOT / "data"; OUT.mkdir(exist_ok=True)
with rasterio.open(SNAP) as s:
    CRS, DST_T, H, W = s.crs, s.transform, s.height, s.width
    xmin, ymin, xmax, ymax = s.bounds
BBOX = list(transform_bounds(CRS, "EPSG:4326", xmin, ymin, xmax, ymax))
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                modifier=planetary_computer.sign_inplace)

def warp_to_grid(hrefs, resampling=Resampling.bilinear, nodata=0):
    """Mosaic one or more COG assets onto the SNAP grid (first valid value wins)."""
    out = np.full((H, W), np.nan, dtype="float32")
    for href in hrefs:
        with rasterio.open(href) as src:
            b = transform_bounds(CRS, src.crs, xmin, ymin, xmax, ymax)
            win = from_bounds(*b, src.transform).round_offsets().round_lengths()
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            arr = src.read(1, window=win).astype("float32")
            nd = src.nodata if src.nodata is not None else nodata
            arr[arr == nd] = np.nan
            tmp = np.full((H, W), np.nan, dtype="float32")
            reproject(arr, tmp, src_transform=src.window_transform(win), src_crs=src.crs,
                      dst_transform=DST_T, dst_crs=CRS, resampling=resampling,
                      src_nodata=np.nan, dst_nodata=np.nan)
            fill = np.isnan(out) & ~np.isnan(tmp)
            out[fill] = tmp[fill]
    return out

def save(name, arrays):
    np.savez_compressed(OUT / f"{name}.npz", **arrays)
    print("saved", name, {k: (v.shape, float(np.nanmean(v))) for k, v in arrays.items()})

def s2(date):
    items = cat.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime=date,
                       query={"s2:mgrs_tile": {"eq": "30SYJ"}}).item_collection()
    print(date, [i.id for i in items])
    hrefs = lambda b: [i.assets[b].href for i in items]
    d = {b: warp_to_grid(hrefs(b)) for b in ("B03", "B04", "B08", "B11")}
    d["SCL"] = warp_to_grid(hrefs("SCL"), Resampling.nearest)
    return d

if __name__ == "__main__":
    save("s2_2024-10-26_pre", s2("2024-10-26"))
    save("s2_2024-10-31_post", s2("2024-10-31"))
    gsw = cat.search(collections=["jrc-gsw"], bbox=BBOX).item_collection()
    save("jrc_gsw_occurrence", {"occurrence": warp_to_grid([i.assets["occurrence"].href for i in gsw],
                                                         Resampling.nearest, nodata=255)})
