# Data

- **Dataset:** UCI Bank Marketing (dataset 222), file `bank-additional-full.csv`, from https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Acquire:** `python scripts/acquire_data.py` (prints the SHA-256; paste it below). Raw data is git-ignored and must not be redistributed.
- **Expected shape:** 41,188 rows, semicolon-separated, 21 columns incl. `y` — verify after download.
- **Access date / SHA-256:** *(fill)*
- **Exclusions & preprocessing:** see `docs/data_card.md` and `configs/data.yaml` (`duration` dropped; positional 60/20/20 split; fixed bins).
- **Processed outputs** (`data/processed/{train,val,test}.csv`) are derived data: share only if licence/privacy allow.
