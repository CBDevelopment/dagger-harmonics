import pickle
import numpy as np
from pprint import pprint

DATA_PATH = r"D:\projects\research\SETI\dagger-harmonics\data\val_data_2010.p"

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

record = data[0]

print("=" * 60)
print("TOP-LEVEL KEYS")
print("=" * 60)
pprint(list(record.keys()))

print("\n" + "=" * 60)
print("SHAPES & TYPES")
print("=" * 60)
for key, val in record.items():
    if isinstance(val, np.ndarray):
        print(f"  {key:20s} ndarray  shape={val.shape}  dtype={val.dtype}")
    elif isinstance(val, tuple):
        print(f"  {key:20s} tuple    len={len(val)}")
        for i, v in enumerate(val):
            v = np.asarray(v)
            print(f"    [{i}]               ndarray  shape={v.shape}  dtype={v.dtype}")
    else:
        print(f"  {key:20s} {type(val).__name__}")

print("\n" + "=" * 60)
print("past_supermag — first record, all stations, all columns")
print("(showing first 10 stations)")
print("=" * 60)
ps = np.asarray(record["past_supermag"])
print(f"  shape: {ps.shape}")
print(f"  last timestep, first 10 stations:")
last = ps[-1] if ps.ndim == 3 else ps
for i, row in enumerate(last[:10]):
    print(f"  station {i:3d}: {row}")

print("\n" + "=" * 60)
print("future_supermag — first record")
print("=" * 60)
fs = np.asarray(record["future_supermag"])
print(f"  shape: {fs.shape}")
last = fs[-1] if fs.ndim == 3 else fs
print("  first 10 stations:")
for i, row in enumerate(last[:10]):
    print(f"  station {i:3d}: {row}")

print("\n" + "=" * 60)
print("past_omni — first record")
print("=" * 60)
po = np.asarray(record["past_omni"])
print(f"  shape: {po.shape}")
print(f"  first timestep: {po[0]}")
print(f"  last  timestep: {po[-1]}")

print("\n" + "=" * 60)
print("coords_radians — first record")
print("=" * 60)
c0 = np.asarray(record["coords_radians"][0]).ravel()
c1 = np.asarray(record["coords_radians"][1]).ravel()
print(
    f"  coords[0] (MLT-derived):  min={np.nanmin(c0):.4f}  max={np.nanmax(c0):.4f}  "
    f"range_deg=[{np.degrees(np.nanmin(c0)):.1f}, {np.degrees(np.nanmax(c0)):.1f}]"
)
print(
    f"  coords[1] (colat):        min={np.nanmin(c1):.4f}  max={np.nanmax(c1):.4f}  "
    f"range_deg=[{np.degrees(np.nanmin(c1)):.1f}, {np.degrees(np.nanmax(c1)):.1f}]"
)

print("\n" + "=" * 60)
print("DATES")
print("=" * 60)
print(f"  past_dates  shape: {np.asarray(record['past_dates']).shape}")
print(f"  past_dates  first: {record['past_dates'].flat[0]}")
print(f"  past_dates  last:  {record['past_dates'].flat[-1]}")
print(f"  future_dates shape: {np.asarray(record['future_dates']).shape}")
print(f"  future_dates first: {record['future_dates'].flat[0]}")

print("\n" + "=" * 60)
print("VALUE RANGES across first 100 records")
print("(helps identify what each past_supermag column contains)")
print("=" * 60)
cols = [[] for _ in range(np.asarray(data[0]["past_supermag"]).shape[-1])]
for record in data[:100]:
    ps = np.asarray(record["past_supermag"])
    last = ps[-1] if ps.ndim == 3 else ps
    for i in range(last.shape[-1]):
        vals = last[:, i]
        cols[i].extend(vals[np.isfinite(vals)].tolist())

print(f"  {'col':>4}  {'min':>10}  {'max':>10}  {'mean':>10}  {'always>=0':>10}")
for i, c in enumerate(cols):
    arr = np.array(c)
    print(
        f"  {i:>4}  {arr.min():>10.3f}  {arr.max():>10.3f}  "
        f"{arr.mean():>10.3f}  {str(bool(np.all(arr >= 0))):>10}"
    )
