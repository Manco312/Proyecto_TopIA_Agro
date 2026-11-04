"""El juez de M2 (LLM-as-a-judge por valor esperado), sin cambios de rúbrica ni de método.

Un solo `forward`; se toma la distribución del siguiente token restringida a '1'..'5' y se
reporta el **valor esperado** (continuo en [1,5]) y la entropía. Determinista.

Vive en el paquete porque el bench de M5 lo necesita para que su scorecard sea comparable
con los de M2, M3 y M4. **No se usa en producción**: el Space no juzga sus propias
respuestas.
"""
from __future__ import annotations

import torch

from ._rubricas import RUBRICA_ADV, RUBRICA_GOLD

_SYS_JUEZ = ("Eres un evaluador agronomico estricto y objetivo. "
             "La extension de la respuesta no es un criterio de calidad.")
_DIG_CACHE = {}


def _ids_digitos(tok):
    k = id(tok)
    if k not in _DIG_CACHE:
        _DIG_CACHE[k] = [tok(str(d), add_special_tokens=False).input_ids[-1] for d in range(1, 6)]
    return _DIG_CACHE[k]


def _prompt_juez(caso, respuesta):
    es_adv = caso.get("tipo") == "adversarial"
    partes = [RUBRICA_ADV if es_adv else RUBRICA_GOLD, "", f"Pregunta: {caso['input']}"]
    if caso.get("criterio"):
        partes.append(f"Criterio de acierto para ESTE caso: {caso['criterio']}")
    if not es_adv and caso.get("esperado"):
        partes.append(f"Respuesta de referencia (guia, no literal): {caso['esperado']}")
    partes += ["", f"Respuesta a evaluar: {respuesta}", "",
               "Responde SOLO con un digito del 1 al 5. Sin explicacion."]
    return _SYS_JUEZ, "\n".join(partes)


@torch.no_grad()
def juez_puntua(caso, respuesta):
    """-> (puntaje continuo 1-5, entropía, distribución)."""
    from .modelos import planificador            # el juez de M2 y el planificador de M3
    tok, model, _ = planificador()               # son el MISMO checkpoint (Qwen2.5-1.5B)
    system, user = _prompt_juez(caso, respuesta)
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    logits = model(**ids).logits[0, -1].float()
    p = torch.softmax(logits[_ids_digitos(tok)], dim=-1)
    escala = torch.arange(1, 6, dtype=p.dtype, device=p.device)
    return (float((p * escala).sum()),
            float(-(p * p.clamp_min(1e-9).log()).sum()),
            [round(float(x), 3) for x in p])


def sim_embeddings(a, b):
    import numpy as np
    from .modelos import embebedor
    ea, eb = embebedor().encode([a, b], normalize_embeddings=True)
    return float(np.dot(ea, eb))
