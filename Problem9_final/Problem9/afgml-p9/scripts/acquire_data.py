"""Download UCI Bank Marketing (dataset 222) and extract bank-additional-full.csv.
You must accept the UCI terms (CC BY 4.0, Moro et al.). Records the SHA-256 for data/README.md."""
import hashlib, io, os, sys, urllib.request, zipfile

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DEST = "data/raw"; os.makedirs(DEST, exist_ok=True)
try:
    blob = urllib.request.urlopen(URL, timeout=60).read()
except Exception as e:
    sys.exit(f"Download failed ({e}). Get the file manually from https://archive.ics.uci.edu/dataset/222/bank+marketing "
             f"and place bank-additional-full.csv in {DEST}/.")
outer = zipfile.ZipFile(io.BytesIO(blob))
inner_name = next((n for n in outer.namelist() if n.endswith("bank-additional.zip")), None)
zf = zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) if inner_name else outer
member = next(n for n in zf.namelist() if n.endswith("bank-additional-full.csv"))
data = zf.read(member); path = os.path.join(DEST, "bank-additional-full.csv"); open(path, "wb").write(data)
print("wrote", path, len(data), "bytes; sha256 =", hashlib.sha256(data).hexdigest())
