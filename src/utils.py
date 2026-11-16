import re, time, unicodedata
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def detect_columns(df):
    """Deteta a coluna de texto (maior comprimento médio) e a de rótulo (menos valores únicos)."""
    obj = [c for c in df.columns if df[c].dtype == object and c != "source_file"]
    if len(obj) < 2:
        raise ValueError("Preciso de pelo menos 2 colunas de texto; define TEXT_COL/LABEL_COL manualmente.")
    text_col = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
    label_col = min([c for c in obj if c != text_col], key=lambda c: df[c].nunique())
    return text_col, label_col


def clean_text(t):
    """Normalização leve: remove URLs, dígitos e espaços repetidos; minúsculas.
    Mantém pontuação e acentos: são pistas fortes para identificar idiomas."""
    t = str(t)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def dominant_script(text):
    """Script Unicode dominante (LATIN, CYRILLIC, ARABIC, ...)."""
    counts = {}
    for ch in str(text):
        if ch.isalpha():
            try:
                s = unicodedata.name(ch).split()[0]
            except ValueError:
                s = "UNKNOWN"
            counts[s] = counts.get(s, 0) + 1
    return max(counts, key=counts.get) if counts else "NONE"


def text_features(s):
    """Features numéricas simples para EDA multivariada."""
    s = s.astype(str)
    out = pd.DataFrame(index=s.index)
    out["n_chars"] = s.str.len()
    out["n_words"] = s.str.split().str.len()
    out["avg_word_len"] = out["n_chars"] / out["n_words"].replace(0, np.nan)
    out["n_digits"] = s.str.count(r"\d")
    out["n_punct"] = s.str.count(r"[^\w\s]")
    out["n_upper"] = s.str.count(r"[A-Z]")
    out["non_ascii_ratio"] = s.apply(lambda x: sum(ord(c) > 127 for c in x) / max(len(x), 1))
    return out


def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def evaluate_model(name, model, Xtr, ytr, Xva, yva):
    """Treina, mede tempos e devolve métricas de validação."""
    t0 = time.time(); model.fit(Xtr, ytr); fit_t = time.time() - t0
    t0 = time.time(); pred = model.predict(Xva); pred_t = time.time() - t0
    return {"model": name,
            "accuracy": accuracy_score(yva, pred),
            "f1_macro": f1_score(yva, pred, average="macro"),
            "fit_time_s": round(fit_t, 2),
            "pred_time_s": round(pred_t, 2)}
