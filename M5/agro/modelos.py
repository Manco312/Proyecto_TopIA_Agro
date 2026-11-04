"""Carga de modelos: una sola vez, perezosa, y con los cuatro backends del redactor.

Regla de producción de la S15: **cargar una vez, atender muchas**. Todo lo que hay aquí son
singletons; `app.py` los calienta al arrancar para que la primera consulta no pague la carga.

Los backends del redactor son el objeto de la técnica de eficiencia de M5:

    fp16_gpu  el sistema de M1-M4 tal cual (base fp16 + adaptador LoRA sin fusionar)
    fp32_cpu  el mismo, sin cuantizar, en CPU -- el "antes" honesto del Space
    int8_cpu  cuantización dinámica int8 de los Linear del transformer   <- se despliega
    int4_gpu  NF4 de bitsandbytes (la técnica del lab S15; exige CUDA)
    groq      API externa; NO es el sistema del curso (en M3 Groq fue el tercer juez)
"""
from __future__ import annotations

import os
import time

import torch

from . import config

_cache = {}


def _from_pretrained(cls, *args, **kw):
    """transformers 5.x usa `dtype=`; las 4.x usaban `torch_dtype=`."""
    try:
        return cls.from_pretrained(*args, **kw)
    except TypeError:
        if "dtype" in kw:
            kw["torch_dtype"] = kw.pop("dtype")
            return cls.from_pretrained(*args, **kw)
        raise


# --------------------------------------------------------------------------------------
# ZeroGPU: construir en CPU al arrancar, mover a la GPU en la primera petición.
#
# `spaces` parchea torch para que `torch.cuda.is_available()` diga True ya al arrancar,
# pero la GPU no está asignada hasta que corre una función decorada con `@spaces.GPU`.
# La mayoría de las rutas de carga sobreviven a eso porque `spaces` las difiere; dos no:
#
#   - `PeftModel.from_pretrained` infiere el dispositivo por su cuenta y le pasa `cuda` a
#     `safetensors.load_file`, que hace una reserva REAL -> "No CUDA GPUs are available";
#   - `from_pretrained(..., device_map={"": 0})` hace lo mismo.
#
# Así que aquí nada toca la GPU durante el arranque: los modelos se arman en CPU, se
# apuntan en `_PENDIENTES_GPU`, y `asegurar_gpu()` los mueve en la primera generación,
# que siempre ocurre dentro de la función decorada.
# --------------------------------------------------------------------------------------
_PENDIENTES_GPU = []


def _peft_en_cpu(base, ruta):
    """Carga el adaptador LoRA en CPU, pase lo que pase con la detección de dispositivo."""
    from peft import PeftModel
    try:
        return PeftModel.from_pretrained(base, ruta, torch_device="cpu")
    except TypeError:                      # versiones de peft sin ese argumento
        return PeftModel.from_pretrained(base, ruta)


def asegurar_gpu():
    """Mueve a la GPU lo que se construyó en CPU. Sólo válido DENTRO de @spaces.GPU."""
    if not _PENDIENTES_GPU:
        return True
    try:
        for m in _PENDIENTES_GPU:
            m.to("cuda")
        _PENDIENTES_GPU.clear()
        return True
    except Exception as e:                                                # noqa: BLE001
        print(f"AVISO: no se pudo mover el modelo a la GPU ({type(e).__name__}: {e}); "
              f"se sigue en CPU.")
        return False


def fijar_semillas(seed=config.SEED):
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:                                   # en ZeroGPU no hay GPU asignada al arrancar
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# --------------------------------------------------------------------------------------
# Recuperación y visión
# --------------------------------------------------------------------------------------
def embebedor():
    if "st" not in _cache:
        from sentence_transformers import SentenceTransformer
        _cache["st"] = SentenceTransformer(config.EMBED_ID, device="cpu")
    return _cache["st"]


def reranker():
    if "rerank" not in _cache:
        from sentence_transformers import CrossEncoder
        _cache["rerank"] = CrossEncoder(config.RERANK_ID, max_length=512, device="cpu")
    return _cache["rerank"]


def clip():
    if "clip" not in _cache:
        from transformers import CLIPModel, CLIPProcessor
        modelo = _from_pretrained(CLIPModel, config.CLIP_ID).to("cpu").eval()
        _cache["clip"] = (modelo, CLIPProcessor.from_pretrained(config.CLIP_ID))
    return _cache["clip"]


# --------------------------------------------------------------------------------------
# El redactor (Qwen2.5-0.5B + LoRA de M1) y sus backends
# --------------------------------------------------------------------------------------
_CORTE_RESPUESTA = "Pregunta:"


def _cortar(texto):
    i = texto.find(_CORTE_RESPUESTA)
    return (texto[:i] if i > 0 else texto).strip()


def _tamano_bytes(modelo):
    """Bytes reales que ocupan los pesos, contando los tensores cuantizados."""
    total = 0
    for p in modelo.parameters():
        total += p.numel() * p.element_size()
    for b in modelo.buffers():
        total += b.numel() * b.element_size()
    for m in modelo.modules():                      # Linear dinámicos: el peso es un atributo
        w = getattr(m, "_packed_params", None)
        if w is not None:
            try:
                qw, _ = m._packed_params._weight_bias()
                total += qw.numel() * qw.element_size()
            except Exception:
                pass
    return total


class RedactorLocal:
    """El generador CRUDO de M1: completion, sin chat template, sin system, greedy."""

    def __init__(self, backend):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        self.backend = backend
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_BASE_ID)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        self.fusionado = False
        self.capas_cuantizadas = 0
        self.pendiente_gpu = False

        if backend == "fp16_gpu":
            # Se arma en CPU y se apunta para mover: en ZeroGPU la GPU no existe todavía.
            base = _from_pretrained(AutoModelForCausalLM, config.MODEL_BASE_ID,
                                    dtype=torch.float16)
            self.modelo = _peft_en_cpu(base, config.MODELO_LORA).eval()
            self.dispositivo, self.dtype = "cuda", "float16"
            self.pendiente_gpu = True
            _PENDIENTES_GPU.append(self.modelo)

        elif backend == "int4_gpu":
            from transformers import BitsAndBytesConfig
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=torch.float16)
            base = _from_pretrained(AutoModelForCausalLM, config.MODEL_BASE_ID,
                                    quantization_config=qc, device_map={"": 0})
            self.modelo = _peft_en_cpu(base, config.MODELO_LORA).eval()
            self.dispositivo, self.dtype = "cuda", "nf4 (double quant, cómputo fp16)"

        elif backend == "fp32peft_cpu":
            # El "antes" de la técnica PEFT: el adaptador SIN fusionar, como lo entregó M1.
            # Cada capa atendida paga dos matmul extra (lora_A y lora_B) por cada proyección.
            base = _from_pretrained(AutoModelForCausalLM, config.MODEL_BASE_ID,
                                    dtype=torch.float32).to("cpu")
            self.modelo = _peft_en_cpu(base, config.MODELO_LORA).eval()
            self.dispositivo, self.dtype = "cpu", "float32 (LoRA sin fusionar)"

        elif backend in ("fp32_cpu", "bf16_cpu", "int8_cpu"):
            # El adaptador se FUSIONA antes de cuantizar: en CPU no tiene sentido pagar dos
            # matmul por capa, y fusionar en fp32 es exacto (W + BA, sin pérdida de rango).
            base = _from_pretrained(AutoModelForCausalLM, config.MODEL_BASE_ID,
                                    dtype=torch.float32).to("cpu")
            self.modelo = _peft_en_cpu(base, config.MODELO_LORA).merge_and_unload().eval()
            self.fusionado = True
            self.dispositivo, self.dtype = "cpu", "float32"
            if backend == "bf16_cpu":
                # El escalón más conservador de la escala fp32 -> fp16 -> int8 -> int4:
                # bfloat16 conserva el MISMO rango de exponente que fp32 y sólo pierde
                # mantisa, que es justo lo que no les importa a los outliers que rompen int8.
                self.modelo = self.modelo.to(torch.bfloat16)
                self.dtype = "bfloat16"
            elif backend == "int8_cpu":
                self.modelo, self.capas_cuantizadas = _cuantizar_int8(self.modelo)
                self.dtype = "qint8 dinámico (Linear del transformer; lm_head en fp32)"
        else:
            raise ValueError(f"backend desconocido: {backend!r}")

        self.segundos_carga = round(time.time() - t0, 2)
        self.bytes_pesos = _tamano_bytes(self.modelo)
        self.parametros = sum(p.numel() for p in self.modelo.parameters())

    def _asegurar_dispositivo(self):
        """Primera generación en ZeroGPU: aquí sí hay GPU asignada."""
        if not self.pendiente_gpu:
            return
        self.pendiente_gpu = False
        if asegurar_gpu():
            return
        # La GPU no apareció: fp16 en CPU va lentísimo, así que se pasa a fp32.
        self.modelo = self.modelo.float()
        self.dispositivo = "cpu"
        self.dtype = "float32 (degradado: la GPU no se asignó)"

    @torch.no_grad()
    def generar(self, prompt, max_new=None):
        max_new = max_new or config.MAX_NEW_SISTEMA
        self._asegurar_dispositivo()
        ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        ids = {k: v.to(self.dispositivo) for k, v in ids.items()}
        kw = dict(max_new_tokens=max_new, do_sample=False,
                  pad_token_id=self.tokenizer.pad_token_id)
        try:
            out = self.modelo.generate(**ids, **kw, stop_strings=[_CORTE_RESPUESTA],
                                       tokenizer=self.tokenizer)
        except (TypeError, ValueError):
            out = self.modelo.generate(**ids, **kw)
        n_in = ids["input_ids"].shape[1]
        self.ultimos_tokens = int(out.shape[1] - n_in)
        return _cortar(self.tokenizer.decode(out[0][n_in:], skip_special_tokens=True))

    def info(self):
        return {"backend": self.backend, "modelo": config.MODEL_BASE_ID,
                "lora": os.path.basename(config.MODELO_LORA), "lora_fusionado": self.fusionado,
                "dispositivo": self.dispositivo, "dtype": self.dtype,
                "parametros": self.parametros, "mib_pesos": round(self.bytes_pesos / 2**20, 1),
                "capas_cuantizadas": self.capas_cuantizadas,
                "pendiente_gpu": self.pendiente_gpu,
                "segundos_carga": self.segundos_carga}


def _cuantizar_int8(modelo):
    """Cuantización dinámica int8 sobre los Linear del transformer.

    `lm_head` se deja en fp32 a propósito: es la capa que decide el token y es donde la
    cuantización dinámica suele costar calidad. El ahorro ya lo da el cuerpo del transformer.

    Dos cuidados de memoria, los dos aprendidos a golpes en la máquina de medición:

    - `inplace=True`. Con la copia por defecto hay un momento en que coexisten el modelo
      fp32 (1.9 GB) y su duplicado, y el proceso muere sin excepción antes de terminar de
      cuantizar. Cuantizar *para* ahorrar memoria no puede necesitar el doble de memoria.
    - El *weight tying*: Qwen2.5-0.5B comparte los pesos de `lm_head` con los del embedding
      (151 936 x 896 = 545 MB en fp32). Al reponer la cabeza en fp32 hay que **volver a
      atarla** al embedding; si se deja suelta, la variante "más ligera" pesa 545 MB más.
    """
    import torch.nn as nn
    objetivo = {nombre for nombre, m in modelo.named_modules()
                if isinstance(m, nn.Linear) and not nombre.endswith("lm_head")}
    q = torch.ao.quantization.quantize_dynamic(
        modelo, {nn.Linear}, dtype=torch.qint8, inplace=True)
    if hasattr(q, "lm_head"):
        emb = q.get_input_embeddings()
        cabeza = nn.Linear(emb.weight.shape[1], emb.weight.shape[0], bias=False)
        cabeza.weight = emb.weight                   # atada: una sola copia de la matriz
        q.lm_head = cabeza
    return q.eval(), len(objetivo)


class RedactorGroq:
    """El redactor por API (plan gratuito de Groq) — el que se despliega.

    Por qué: desde M3 el cuello de botella medido de este sistema **no es el conocimiento
    sino la generación**. El retrieval acierta (hit@3 = 12/12 en las fichas externas) y aun
    así el Qwen2.5-0.5B afinado en M1 no convierte el contexto en una recomendación
    utilizable. M4 lo confirmó por el otro lado: con el diagnóstico correcto puesto en el
    prompt, el modelo igual recetó sobre una hoja sana. Cambiar sólo esa pieza es la
    intervención más barata que ataca el fallo medido.

    Qué NO cambia: el diagnóstico lo sigue dando el clasificador de M4, la evidencia la
    sigue trayendo el índice de M3, y el **filtro de salida de M5 se aplica igual** — un
    modelo de 120B es más convincente, y por tanto más peligroso, cuando inventa una dosis.

    Tres cuidados de producción, los tres del enunciado de despliegue del curso:
      - el plan gratuito limita peticiones por minuto y tokens por día -> reintentos con
        espera y un mensaje claro en vez de una traza;
      - los casos ya ensayados se guardan en caché, para que la demo no dependa de la red;
      - la clave se lee del entorno (Secrets del Space), nunca del código.
    """

    # Si Groq retira el modelo configurado (en M3 pasó con llama-3.3-70b: 404
    # model_not_found y el error quedaba oculto), se prueba el siguiente de la lista.
    ALTERNATIVOS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]

    def __init__(self, modelo=None):
        from groq import Groq
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY no configurada. En el Space: Settings -> Variables and "
                "secrets -> New secret. En local: M5/.env con GROQ_API_KEY=...")
        self.cliente = Groq(api_key=config.GROQ_API_KEY, timeout=45.0, max_retries=0)
        self.backend, self.dispositivo = "groq", "api"
        self.modelo_id = modelo or config.GROQ_MODELO
        self.dtype, self.segundos_carga = self.modelo_id, 0.0
        self.bytes_pesos, self.parametros, self.capas_cuantizadas = 0, 0, 0
        self.fusionado = False
        self.ultimos_tokens = 0
        self.llamadas, self.fallos, self.aciertos_cache, self.truncadas = 0, 0, 0, 0
        self.ultimo_error, self.ultimo_finish = None, None
        self.max_new_por_defecto = config.MAX_NEW_GROQ
        self._cache = {}

    @staticmethod
    def _partir(prompt):
        """El prompt de M3 es una completion (`SYSTEM_RAG` + Contexto + Pregunta +
        'Respuesta:'). Un modelo de chat rinde mucho mejor si esa instrucción va como
        `system` y el resto como `user`, así que se parte aquí en vez de mandarlo crudo.
        El texto es EL MISMO: sólo cambia el reparto entre roles."""
        from .herramientas import SYSTEM_RAG
        if prompt.startswith(SYSTEM_RAG):
            cuerpo = prompt[len(SYSTEM_RAG):].lstrip()
        else:
            cuerpo = prompt
        cuerpo = cuerpo.removesuffix("Respuesta:").rstrip()
        return SYSTEM_RAG, cuerpo

    def generar(self, prompt, max_new=None):
        system, user = self._partir(prompt)
        return self.chat(system, user, max_new)

    def chat(self, system, user, max_new=None):
        """Una llamada con reintentos, caché y cambio de modelo si Groq retiró el actual.
        La usan el redactor y, si así se configura, el planificador ReAct."""
        max_new = max_new or config.MAX_NEW_GROQ
        clave = (hash(system), hash(user), max_new)
        if clave in self._cache:                       # la demo no depende de la red
            self.aciertos_cache += 1
            return self._cache[clave]

        espera, ultimo = 2.0, None
        for intento in range(3):
            try:
                self.llamadas += 1
                r = self.cliente.chat.completions.create(
                    model=self.modelo_id, temperature=0, top_p=1,
                    max_completion_tokens=max_new,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                self.ultimos_tokens = getattr(getattr(r, "usage", None),
                                              "completion_tokens", 0) or 0
                # "length" = se acabaron los tokens antes de que el modelo terminara.
                # Es un síntoma operativo, así que se registra en vez de pasarlo por alto.
                self.ultimo_finish = getattr(r.choices[0], "finish_reason", None)
                if self.ultimo_finish == "length":
                    self.truncadas += 1
                    print(f"AVISO: respuesta truncada en {max_new} tokens "
                          f"(sube AGRO_MAX_TOKENS)")
                texto = _cortar(r.choices[0].message.content or "")
                self._cache[clave] = texto
                return texto
            except Exception as e:                                        # noqa: BLE001
                ultimo = e
                self.fallos += 1
                self.ultimo_error = f"{type(e).__name__}: {e}"[:200]
                msg = str(e).lower()
                if "model_not_found" in msg or "does not exist" in msg:
                    siguiente = next((m for m in self.ALTERNATIVOS if m != self.modelo_id),
                                     None)
                    if siguiente:
                        print(f"AVISO: Groq no tiene {self.modelo_id}; se pasa a {siguiente}")
                        self.modelo_id = self.dtype = siguiente
                        continue
                if intento < 2 and ("rate" in msg or "429" in msg or "timeout" in msg
                                    or "connection" in msg):
                    time.sleep(espera)
                    espera *= 2
                    continue
                break
        raise RuntimeError(
            "El redactor por API no respondió (plan gratuito de Groq: hay límite de "
            f"peticiones por minuto y de tokens por día). Detalle: {ultimo}")

    def info(self):
        return {"backend": "groq", "modelo": self.modelo_id, "dispositivo": "api",
                "dtype": "n/a (API)", "parametros": 0, "mib_pesos": 0.0,
                "lora_fusionado": False, "capas_cuantizadas": 0, "pendiente_gpu": False,
                "segundos_carga": 0.0, "llamadas": self.llamadas, "fallos": self.fallos,
                "aciertos_cache": self.aciertos_cache, "ultimo_error": self.ultimo_error,
                "max_tokens": self.max_new_por_defecto, "truncadas": self.truncadas,
                "ultimo_finish_reason": self.ultimo_finish}


def redactor(backend=None):
    backend = config.resolver_backend(backend)
    clave = f"redactor:{backend}"
    if clave not in _cache:
        _cache[clave] = RedactorGroq() if backend == "groq" else RedactorLocal(backend)
    return _cache[clave]


def soltar_redactor(backend):
    """Para el bench: liberar una variante antes de cargar la siguiente."""
    import gc
    _cache.pop(f"redactor:{backend}", None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# El planificador ReAct de M3 §14 (Qwen2.5-1.5B). Sólo se carga si la ruta es "react".
# --------------------------------------------------------------------------------------
def planificador():
    """Qwen2.5-1.5B: el "cerebro" ReAct de M3 §14 y el juez de M2 §4 son el mismo checkpoint.

    Se carga con `device_map` cuando hay CUDA. Con `.from_pretrained(...).to("cuda")` los
    3.1 GB se materializan primero en RAM y en una máquina con poca memoria libre el proceso
    muere sin excepción ("el archivo de paginación es demasiado pequeño", os error 1455);
    con `device_map` los pesos van al dispositivo shard a shard y el pico en RAM es mínimo.
    """
    if "plan" not in _cache:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hay_cuda = torch.cuda.is_available()
        dev = "cuda" if hay_cuda else "cpu"
        tok = AutoTokenizer.from_pretrained(config.PLANIFICADOR_ID)

        if config.en_zerogpu():
            # Igual que el redactor: en CPU al arrancar, a la GPU en la primera petición.
            # `device_map={"": 0}` reservaría memoria de GPU aquí mismo y el Space no
            # llegaría ni a levantar.
            mod = _from_pretrained(AutoModelForCausalLM, config.PLANIFICADOR_ID,
                                   dtype=torch.float16).eval()
            _PENDIENTES_GPU.append(mod)
            _cache["plan"] = (tok, mod, "cuda")
            return _cache["plan"]

        intentos = ([dict(dtype=torch.float16, device_map={"": 0}),
                     dict(dtype=torch.float16)] if hay_cuda else []) + [dict(dtype="auto")]
        mod = None
        for i, kw in enumerate(intentos):
            try:
                mod = _from_pretrained(AutoModelForCausalLM, config.PLANIFICADOR_ID, **kw)
                if "device_map" not in kw and hay_cuda and i < len(intentos) - 1:
                    mod = mod.to("cuda")
                break
            except Exception as e:             # OOM de CUDA, o de RAM al hacer el .to()
                print(f"   juez/planificador: intento {i + 1} falló "
                      f"({type(e).__name__}); probando el siguiente")
                import gc
                gc.collect()
                if hay_cuda:
                    torch.cuda.empty_cache()
        if mod is None:
            raise RuntimeError("no se pudo cargar el planificador/juez en ningún dispositivo")
        dev = next(mod.parameters()).device.type
        _cache["plan"] = (tok, mod.eval(), dev)
    return _cache["plan"]


def generar_agente(system, user, max_new_tokens=120):
    """El "cerebro" del bucle ReAct. Groq si es el motor desplegado, Qwen2.5-1.5B si no.

    Que el planificador y el redactor compartan motor no es casualidad: con la API no hace
    falta cargar un 1.5B sólo para elegir herramientas, que era lo que hacía inviable la
    ruta ReAct en CPU (ver la ablación del planificador en el reporte).
    """
    if config.resolver_motor_planificador() == "groq":
        return redactor("groq").chat(system, user, max_new_tokens)
    return generar_planificador(system, user, max_new_tokens)


@torch.no_grad()
def generar_planificador(system, user, max_new_tokens=120):
    tok, mod, dev = planificador()
    if _PENDIENTES_GPU:                     # ZeroGPU: aquí sí hay GPU asignada
        if not asegurar_gpu():
            mod.float()
    dev = next(mod.parameters()).device
    prompt = tok.apply_chat_template([{"role": "system", "content": system},
                                      {"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(dev)
    out = mod.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def revision(model_id):
    """Commit hash del snapshot local, para reproducibilidad exacta (igual que M3/M4)."""
    try:
        from huggingface_hub import snapshot_download
        p = snapshot_download(model_id, local_files_only=True)
        return os.path.basename(p) or os.path.basename(os.path.dirname(p))
    except Exception:
        return "(desconocida)"
