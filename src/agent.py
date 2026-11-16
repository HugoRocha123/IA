import os, json
import numpy as np, joblib
from collections import Counter
from utils import clean_text


class ClassifierAgent:
    """Skill: classificar textos e devolver confiança + top-3."""
    def __init__(self, bundle_path="models/best_classifier.joblib", low_conf=0.5):
        b = joblib.load(bundle_path)
        self.pipe, self.le, self.meta, self.low_conf = b["pipeline"], b["label_encoder"], b["metadata"], low_conf

    def _proba(self, texts):
        clf, X = self.pipe.named_steps["clf"], self.pipe.named_steps["prep"].transform(texts)
        if hasattr(clf, "predict_proba"):
            return clf.predict_proba(X)
        d = clf.decision_function(X)                       # LinearSVC etc.: softmax das margens
        e = np.exp(d - d.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def predict(self, texts):
        texts = [texts] if isinstance(texts, str) else list(texts)
        cleaned = [clean_text(t) for t in texts]
        P = self._proba(cleaned)
        out = []
        for raw, c, p in zip(texts, cleaned, P):
            idx = np.argsort(p)[::-1][:3]
            conf = float(p[idx[0]])
            out.append({"text": raw, "language": self.le.classes_[idx[0]], "confidence": conf,
                        "top3": [(self.le.classes_[i], round(float(p[i]), 3)) for i in idx],
                        "low_confidence": bool(conf < self.low_conf or len(c) == 0)})
        return out


class ReporterAgent:
    """Agente generativo: transforma estatísticas em relatório. Usa LLM (Transformers) se disponível; senão, template."""
    def __init__(self, use_llm=False, model_id="Qwen/Qwen2.5-0.5B-Instruct"):
        self.gen = None
        if use_llm:
            try:
                from transformers import pipeline
                self.gen = pipeline("text-generation", model=model_id)
            except Exception as e:
                print("LLM indisponível, a usar template:", e)

    @staticmethod
    def stats(preds):
        langs = Counter(p["language"] for p in preds)
        confs = [p["confidence"] for p in preds]
        return {"n": len(preds), "top_languages": langs.most_common(5),
                "mean_confidence": round(float(np.mean(confs)), 3) if confs else 0.0,
                "low_conf_ratio": round(sum(p["low_confidence"] for p in preds) / max(len(preds), 1), 3)}

    def report(self, preds):
        s = self.stats(preds)
        template = (f"Relatório de identificação de idioma\n- Textos analisados: {s['n']}\n"
                    f"- Idiomas mais frequentes: {s['top_languages']}\n- Confiança média: {s['mean_confidence']}\n"
                    f"- % com baixa confiança: {s['low_conf_ratio']:.1%}")
        if self.gen is None:
            return template
        try:
            msgs = [{"role": "system", "content": "És um analista. Escreve um resumo curto em português europeu."},
                    {"role": "user", "content": "Resume estas estatísticas e sugere uma ação:\n" + json.dumps(s, ensure_ascii=False)}]
            return template + "\n\nResumo LLM: " + self.gen(msgs, max_new_tokens=150)[0]["generated_text"][-1]["content"]
        except Exception:
            return template


class NotifierAgent:
    """Skill: enviar mensagens por Telegram (dry-run se não houver token)."""
    def __init__(self, token=None, chat_id=None):
        self.token = token or os.getenv("TELEGRAM_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    def send(self, message):
        if not (self.token and self.chat_id):
            print("[Telegram DRY-RUN]\n" + message); return False
        import requests
        r = requests.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                          data={"chat_id": self.chat_id, "text": message}, timeout=10)
        return r.ok


class Orchestrator:
    """Coordena Classificador -> Repórter -> Notificador e aplica regras de alerta."""
    def __init__(self, classifier, reporter, notifier, max_low_conf=0.25, baseline=None):
        self.c, self.r, self.n = classifier, reporter, notifier
        self.max_low_conf, self.baseline = max_low_conf, baseline

    def process_batch(self, texts):
        preds = self.c.predict(texts)
        st = self.r.stats(preds)
        alerts = []
        if st["low_conf_ratio"] > self.max_low_conf:
            alerts.append(f"⚠️ {st['low_conf_ratio']:.0%} das predições com baixa confiança (limite {self.max_low_conf:.0%})")
        if self.baseline:
            seen = {l for l, _ in st["top_languages"]}
            novel = seen - set(self.baseline)
            if novel:
                alerts.append(f"⚠️ Idiomas fora do baseline: {sorted(novel)}")
        return preds, st, alerts

    def daily_dashboard(self, texts):
        preds, st, alerts = self.process_batch(texts)
        msg = "📊 Dashboard diário\n" + self.r.report(preds) + ("\n\n" + "\n".join(alerts) if alerts else "\n\n✅ Sem alertas")
        self.n.send(msg)
        return msg
