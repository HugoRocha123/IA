def build_contract(text_col, label_col, n_classes):
    """Contrato de dados definido ANTES da modelação."""
    return {
        "dataset": "zarajamshaid/language-identification-datasst",
        "format": "CSV (UTF-8), uma linha por amostra",
        "columns": {
            text_col:  {"type": "string", "unit": "n/a (texto livre)", "nullable": False, "role": "feature"},
            label_col: {"type": "string (categoria)", "unit": "n/a", "nullable": False, "role": "target",
                        "n_classes": n_classes},
        },
        "sampling_frequency": "n/a (não é série temporal)",
        "temporal_component": False,
        "constraints": {"min_text_chars": 1, "max_missing_ratio": 0.01},
    }


def validate_contract(df, contract):
    """Devolve lista de (verificação, passou?, detalhe)."""
    res = []
    for col, spec in contract["columns"].items():
        res.append((f"coluna '{col}' existe", col in df.columns, ""))
        if col in df.columns:
            miss = df[col].isna().mean()
            res.append((f"'{col}' missing <= {contract['constraints']['max_missing_ratio']:.0%}",
                        miss <= contract["constraints"]["max_missing_ratio"], f"{miss:.2%}"))
            if spec["type"].startswith("string"):
                res.append((f"'{col}' é texto", df[col].dropna().map(lambda x: isinstance(x, str)).all(), ""))
    res.append(("sem componente temporal", not contract["temporal_component"], ""))
    return res
