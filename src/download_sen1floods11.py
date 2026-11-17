"""Download the hand-labelled part of Sen1Floods11 (Bonafilia et al., CVPR-W 2020): Sentinel-1 VV/VH chips
(512x512, 10 m) with hand-drawn water labels from 11 flood events, plus the official train/valid/test splits."""
import pathlib, requests, concurrent.futures as cf

def fetch(url, path):
    r = requests.get(url, timeout=120); r.raise_for_status(); pathlib.Path(path).write_bytes(r.content)
B = "https://storage.googleapis.com/sen1floods11/v1.1"
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "sen1floods11"
(OUT / "S1Hand").mkdir(parents=True, exist_ok=True); (OUT / "LabelHand").mkdir(exist_ok=True)
files = []
for split in ("train", "valid", "test"):
    name = f"flood_{split}_data.csv"
    fetch(f"{B}/splits/flood_handlabeled/{name}", OUT / name)
    for line in open(OUT / name).read().split():
        s1, lab = line.strip().split(",")
        files += [("S1Hand", s1), ("LabelHand", lab)]
def get(f):
    d, n = f; p = OUT / d / n
    if not p.exists():
        fetch(f"{B}/data/flood_events/HandLabeled/{d}/{n}", p)
with cf.ThreadPoolExecutor(16) as ex: list(ex.map(get, files))
print(len(files) // 2, "chip pairs in", OUT)
