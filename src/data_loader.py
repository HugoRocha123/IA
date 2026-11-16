import os, glob
import pandas as pd

DATASET = "zarajamshaid/language-identification-datasst"


def load_raw(dataset=DATASET):
    """Descarrega (kagglehub, com cache) e junta todos os CSV com o mesmo esquema."""
    import kagglehub
    path = kagglehub.dataset_download(dataset)
    files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    print("Ficheiros CSV encontrados:", [os.path.basename(f) for f in files])
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except UnicodeDecodeError:
            d = pd.read_csv(f, encoding="latin-1")
        d = d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]
        d["source_file"] = os.path.basename(f)
        frames.append(d)
    base_cols = set(frames[0].columns) - {"source_file"}
    frames = [d for d in frames if set(d.columns) - {"source_file"} == base_cols]
    return pd.concat(frames, ignore_index=True)
