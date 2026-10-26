"""Funciones comunes de la Tarea 2: OCR, métricas, seudonimización, LLM y gráficos.

Los dos notebooks importan este módulo para que la misma lógica (por ejemplo, la
normalización de textos antes de CER/WER) se use igual en todos los modelos.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "datos"
RAW = DATOS / "raw"
EJEMPLOS = DATOS / "ejemplos"
FIGURAS = RAIZ / "figuras"
MODELOS = RAIZ / "modelos"


# ═══════════════════════════════════════════════ OCR
OPCIONES_BASE = dict(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
OPCIONES_MEJORADO = dict(use_doc_orientation_classify=True, use_doc_unwarping=True, use_textline_orientation=True)


def preparar_imagen(ruta: Path, destino_dir: Path, lado_max=2000) -> Path:
    """Preprocesamiento mínimo y documentado: RGB, rotación EXIF y lado mayor <= 2000 px.

    No se binariza ni se ajusta contraste: la línea base debe ser PaddleOCR "tal cual".
    """
    from PIL import Image, ImageOps
    destino_dir.mkdir(parents=True, exist_ok=True)
    destino = destino_dir / (Path(ruta).stem + ".png")
    if not destino.exists():
        im = ImageOps.exif_transpose(Image.open(ruta)).convert("RGB")
        if max(im.size) > lado_max:
            im.thumbnail((lado_max, lado_max))
        im.save(destino)
    return destino


def crear_paddle(opciones: dict | None = None, device="cpu"):
    """PaddleOCR 3.7 con PyTorch (engine='transformers', como en la Sesión 10).

    Si el motor PyTorch falla en tu equipo, se usa el motor por defecto de PaddlePaddle.
    Devuelve (modelo, descripción de la configuración).
    """
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR
    import paddleocr
    opciones = dict(OPCIONES_BASE if opciones is None else opciones)
    try:
        modelo = PaddleOCR(engine="transformers", device=device, **opciones)
        motor = "transformers (PyTorch)"
    except Exception as e:  # noqa: BLE001
        print(f"engine='transformers' no disponible ({type(e).__name__}); uso el motor PaddlePaddle")
        modelo = PaddleOCR(device=device, **opciones)
        motor = "paddle"
    activos = [k.replace("use_", "") for k, v in opciones.items() if v]
    desc = f"PaddleOCR {paddleocr.__version__} PP-OCRv6 · motor {motor} · {device} · " + \
           (f"con {', '.join(activos)}" if activos else "sin preprocesamiento extra")
    return modelo, desc


def lineas_vacias() -> pd.DataFrame:
    return pd.DataFrame({"texto": pd.Series(dtype=str), **{c: pd.Series(dtype=float) for c in ["confianza", "x0", "y0", "x1", "y1"]}})


def leer_paddle(modelo, ruta) -> pd.DataFrame:
    """Una fila por línea: texto, confianza y caja (x0, y0, x1, y1)."""
    r = modelo.predict(str(ruta))[0]
    textos = list(r["rec_texts"])
    if not textos:
        return lineas_vacias()
    cajas = np.asarray(r["rec_boxes"]).reshape(-1, 4)
    return pd.DataFrame({"texto": textos, "confianza": np.round(r["rec_scores"], 3),
                         "x0": cajas[:, 0], "y0": cajas[:, 1], "x1": cajas[:, 2], "y1": cajas[:, 3]})


def crear_easyocr():
    import easyocr
    lector = easyocr.Reader(["es"], gpu=False, verbose=False)
    return lector, f"EasyOCR {easyocr.__version__} (CRAFT + CRNN, lang=es, CPU)"


def leer_easyocr(lector, ruta) -> str:
    # paragraph=False y orden por posición: igual criterio de lectura que PaddleOCR (línea a línea)
    res = lector.readtext(str(ruta), detail=1, paragraph=False)
    res = sorted(res, key=lambda r: (round(min(p[1] for p in r[0]) / 20), min(p[0] for p in r[0])))
    return "\n".join(r[1] for r in res)


def tesseract_disponible() -> bool:
    try:
        import pytesseract
        if os.name == "nt" and not os.environ.get("TESSERACT_CMD"):
            for c in [r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]:
                if Path(c).exists():
                    pytesseract.pytesseract.tesseract_cmd = c
        elif os.environ.get("TESSERACT_CMD"):
            pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
        return "spa" in pytesseract.get_languages()
    except Exception:  # noqa: BLE001
        return False


def leer_tesseract(ruta) -> str:
    import pytesseract
    from PIL import Image
    return pytesseract.image_to_string(Image.open(ruta), lang="spa", config="--psm 3")


def texto_de_lineas(lineas: pd.DataFrame) -> str:
    return "\n".join(str(t) for t in lineas["texto"]) if len(lineas) else ""


# ═══════════════════════════════════════════════ métricas (mismo criterio para todos los modelos)
MARCAS = re.compile(r"\[(?:HABLANTE_\d+|H\d+|[A-ZÁÉÍÓÚÑ_]+)\]\s*:?")


def quitar_marcas(texto: str) -> str:
    """Quita [HABLANTE_1], [PERSONA], [DNI]... que no son palabras habladas ni leídas."""
    return MARCAS.sub(" ", str(texto or ""))


def normalizar(texto: str) -> str:
    """Igual que voz.normalizar (Sesión 13): minúsculas, sin puntuación ni espacios repetidos.

    Se conservan tildes y eñes. Además se quitan las marcas entre corchetes.
    """
    texto = unicodedata.normalize("NFC", quitar_marcas(texto).lower())
    texto = re.sub(r"[^\w\s']", " ", texto)
    texto = texto.replace("_", " ")
    return re.sub(r"\s+", " ", texto).strip()


def _ediciones(a, b) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:  # versión lenta en Python puro
        fila = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            nueva = [i]
            for j, cb in enumerate(b, 1):
                nueva.append(min(fila[j] + 1, nueva[j - 1] + 1, fila[j - 1] + (ca != cb)))
            fila = nueva
        return fila[-1]


def cer_uno(ref: str, hip: str) -> float:
    r, h = normalizar(ref), normalizar(hip)
    return _ediciones(r, h) / max(len(r), 1)


def wer_uno(ref: str, hip: str) -> float:
    r, h = normalizar(ref).split(), normalizar(hip).split()
    return _ediciones(r, h) / max(len(r), 1)


def cer_corpus(refs, hips) -> float:
    """CER agregado: total de ediciones / total de caracteres de referencia (como jiwer)."""
    pares = [(normalizar(r), normalizar(h)) for r, h in zip(refs, hips)]
    return sum(_ediciones(r, h) for r, h in pares) / max(sum(len(r) for r, _ in pares), 1)


def wer_corpus(refs, hips) -> float:
    pares = [(normalizar(r).split(), normalizar(h).split()) for r, h in zip(refs, hips)]
    return sum(_ediciones(r, h) for r, h in pares) / max(sum(len(r) for r, _ in pares), 1)


# ═══════════════════════════════════════════════ seudonimización
PATRONES = {
    "DNI": re.compile(r"\b(?:D\.?\s?N\.?\s?I\.?\s*(?:N[°º.o]*\s*)?:?\s*)?\d{8}\b"),
    "RUC": re.compile(r"\b(?:10|15|17|20)\d{9}\b"),
    "TELEFONO": re.compile(r"(?<!\d)(?:\+?51[\s-]?)?9\d{2}[\s-]?\d{3}[\s-]?\d{3}(?!\d)"),
    "CORREO": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
}
NOMBRE = r"[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+(?:de\s+|del\s+|de\s+la\s+)?[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3}"
PRESENTACION = re.compile(rf"(\b(?i:me llamo|mi nombre es|mi nombre|yo soy|soy la señora|soy el señor|la señora|el señor|señora|señor|doña)\s+)({NOMBRE})")


def cargar_seudonimos(ruta: Path | None = None) -> dict[str, str]:
    """Diccionario nombre real -> seudónimo. Vive en datos/raw/ (NO se versiona: tiene nombres reales)."""
    ruta = ruta or (RAW / "seudonimos.csv")
    if not ruta.exists():
        return {}
    t = pd.read_csv(ruta, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    return {r.nombre.strip(): r.seudonimo.strip() for r in t.itertuples() if r.nombre.strip()}


def seudonimizar_texto(texto: str, diccionario: dict[str, str] | None = None, modo="audio") -> tuple[str, dict]:
    """Reemplaza datos personales. Devuelve (texto, conteo por tipo de entidad).

    - Siempre: DNI, RUC, teléfonos celulares y correos.
    - Nombres del diccionario (comparación difusa con voz.seudonimizar, Sesión 13).
    - Solo en audio (personas privadas): nombres tras "me llamo", "señora", etc.
      En documentos históricos los nombres de figuras públicas (p. ej. Ricardo Palma) se
      conservan: no son datos personales de particulares. Esta decisión se documenta.
    """
    conteo = {}
    salida = str(texto or "")
    for etiqueta, patron in PATRONES.items():
        salida, n = patron.subn(f"[{etiqueta}]", salida)
        if n:
            conteo[etiqueta] = n
    if diccionario:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from voz import seudonimizar
        for nombre, seud in diccionario.items():
            antes = salida.count(seud)
            salida = seudonimizar(salida, {nombre: seud})
            n = salida.count(seud) - antes
            if n:
                conteo["NOMBRE_DICCIONARIO"] = conteo.get("NOMBRE_DICCIONARIO", 0) + n
    if modo == "audio":
        salida, n = PRESENTACION.subn(lambda m: m.group(1) + "[PERSONA]", salida)
        if n:
            conteo["PERSONA_PRESENTACION"] = n
    return salida, conteo


# ═══════════════════════════════════════════════ audio: cortar entrevistas largas en clips
def cortar_en_clips(duracion: float, tramos: list[tuple[float, float]], objetivo=75.0, maximo=100.0) -> list[tuple[float, float]]:
    """Agrupa los tramos de voz del VAD en clips de ~`objetivo` s, cortando siempre en un silencio."""
    if duracion <= maximo or not tramos:
        return [(0.0, round(duracion, 2))]
    clips, ini = [], 0.0
    for k, (a, b) in enumerate(tramos):
        sig = tramos[k + 1][0] if k + 1 < len(tramos) else None
        if sig is None:
            break
        if b - ini >= objetivo or (sig - ini) > maximo:
            corte = round((b + sig) / 2, 2)
            clips.append((round(ini, 2), corte))
            ini = corte
    clips.append((round(ini, 2), round(duracion, 2)))
    if len(clips) > 1 and clips[-1][1] - clips[-1][0] < 15:   # el último trozo muy corto se une al anterior
        a, _ = clips[-2]
        clips[-2:] = [(a, clips[-1][1])]
    return clips


ALUCINACIONES = re.compile(r"gracias por ver|suscr[ií]bete|subt[ií]tulos (?:realizados|por)|amara\.org|"
                           r"thanks for watching|música de fondo", re.I)


def posibles_alucinaciones(texto: str) -> list[str]:
    """Frases típicas que Whisper inventa en silencios o música, y repeticiones en bucle."""
    hallado = [m.group(0) for m in ALUCINACIONES.finditer(texto or "")]
    palabras = normalizar(texto).split()
    for n in (3, 4, 5):
        for i in range(len(palabras) - 3 * n):
            g = palabras[i:i + n]
            if g == palabras[i + n:i + 2 * n] == palabras[i + 2 * n:i + 3 * n]:
                hallado.append("repetición: " + " ".join(g))
                break
    return hallado


# ═══════════════════════════════════════════════ LLM: extracción estructurada (Gemini)
CAMPOS = {
    "documento": ["modalidad", "tipo_documento", "tema", "resumen", "emisor", "fecha", "monto", "lugar", "legible"],
    "audio": ["modalidad", "tema", "resumen", "numero_hablantes", "citas_clave", "lugar"],
}

INSTRUCCIONES = """Eres un asistente de archivística. Recibes texto obtenido automáticamente (OCR de un
documento peruano o transcripción automática de un audio en español peruano). Devuelve SOLO un
objeto JSON con exactamente estas claves y ninguna otra:

{esquema}

Reglas obligatorias:
1. Usa únicamente información que aparezca en el texto. No completes con conocimiento externo.
2. Si un dato no aparece de forma explícita, usa null. Nunca inventes emisores, fechas ni montos.
3. "tema": 2 a 6 palabras. "resumen": una o dos oraciones en español, fieles al texto.
4. Fechas tal como aparecen (no conviertas formatos). Montos con su moneda si aparece (p. ej. "500 000 intis").
5. En audio, "citas_clave" es una lista de 1 a 3 frases copiadas literalmente del texto (o []), y
   "numero_hablantes" solo si el texto trae marcas de hablante o lo dice explícitamente; si no, null.
6. En documentos, "legible" es false si el texto es mayormente basura del OCR (palabras sin sentido).
7. El texto puede contener marcas como [PERSONA] o [DNI]: son datos ya anonimizados; no intentes reconstruirlos.
"""

ESQUEMAS = {
    "documento": '{"modalidad": "documento", "tipo_documento": str|null, "tema": str, "resumen": str, '
                 '"emisor": str|null, "fecha": str|null, "monto": str|null, "lugar": str|null, "legible": bool}',
    "audio": '{"modalidad": "audio", "tema": str, "resumen": str, "numero_hablantes": int|null, '
             '"citas_clave": [str], "lugar": str|null}',
}

MODELOS_GEMINI = ["gemini-flash-latest", "gemini-3.6-flash", "gemini-3-flash", "gemini-flash-lite-latest", "gemini-2.5-flash"]


class ExtractorGemini:
    """Extrae campos JSON con Gemini (google-genai). Guarda cada respuesta en caché para no
    repetir llamadas (y no gastar cuota) si el notebook se vuelve a ejecutar."""

    def __init__(self, cache: Path | None = None, modelo: str | None = None, pausa=4.5):
        from google import genai
        try:
            from dotenv import load_dotenv
            load_dotenv(RAIZ / ".env")
        except ImportError:
            pass
        clave = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not clave:
            raise RuntimeError("Falta GEMINI_API_KEY. Crea tarea_2/.env con GEMINI_API_KEY=tu_clave (no lo subas a Git).")
        self.cliente = genai.Client(api_key=clave)
        self.cache_ruta = cache or (RAW / "llm_cache.jsonl")
        self.cache = {}
        if self.cache_ruta.exists():
            for linea in self.cache_ruta.read_text(encoding="utf-8").splitlines():
                if linea.strip():
                    d = json.loads(linea)
                    if d.get("ok", any(v not in (None, [], "documento", "audio") for v in d["campos"].values())):
                        self.cache[d["clave"]] = d      # las respuestas fallidas no se reutilizan
        self.pausa = pausa
        self.disponibles = []
        self.modelo = modelo or os.environ.get("GEMINI_MODEL") or self._elegir_modelo()

    def _elegir_modelo(self):
        """Pregunta a la API qué modelos 'flash' tiene esta cuenta y usa el primero que responda.

        Los nombres cambian con el tiempo (2.5 -> 3.x): por eso no se fija uno a mano. Un 503
        (modelo saturado) es temporal: se reintenta con espera antes de pasar al siguiente.
        """
        candidatos = list(MODELOS_GEMINI)
        try:
            disponibles = []
            for m in self.cliente.models.list():
                nombre = m.name.removeprefix("models/")
                acciones = getattr(m, "supported_actions", None) or ["generateContent"]
                if ("flash" in nombre and "generateContent" in acciones
                        and not re.search(r"image|tts|audio|live|embed|thinking|exp", nombre)):
                    disponibles.append(nombre)
            def orden(n):  # más nuevo primero; estable antes que preview; flash antes que flash-lite
                v = re.search(r"(\d+(?:\.\d+)?)", n)
                return (-(float(v.group(1)) if v else 0), "preview" in n, "lite" in n, n)
            disponibles.sort(key=orden)
            print("Modelos flash disponibles en tu cuenta:", ", ".join(disponibles[:8]) or "(ninguno)")
            candidatos = disponibles + [c for c in candidatos if c not in disponibles]
            self.disponibles = candidatos
        except Exception as e:  # noqa: BLE001
            print(f"  no pude listar modelos ({str(e)[:80]}); pruebo nombres conocidos")
        for m in candidatos:
            for intento in range(3):
                try:
                    self.cliente.models.generate_content(model=m, contents="Responde solo: ok")
                    return m
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    if ("503" in msg or "UNAVAILABLE" in msg) and intento < 2:
                        print(f"  {m}: saturado, reintento en {15 * (intento + 1)} s")
                        time.sleep(15 * (intento + 1))
                        continue
                    print(f"  modelo {m} no disponible: {msg[:90]}")
                    break
        raise RuntimeError("Ningún modelo Gemini respondió; define GEMINI_MODEL en .env")

    def _siguiente_modelo(self):
        """Pasa al siguiente modelo flash si el actual se quedó sin cuota (429) o no existe (404)."""
        if not self.disponibles:
            try:
                self._elegir_modelo()
            except Exception:  # noqa: BLE001
                pass
        resto = [m for m in self.disponibles if m != self.modelo and m not in getattr(self, "_agotados", set())]
        self._agotados = getattr(self, "_agotados", set()) | {self.modelo}
        if not resto:
            return False
        print(f"  → cambio de modelo: {self.modelo} → {resto[0]}")
        self.modelo = resto[0]
        return True

    def extraer(self, id_: str, modalidad: str, texto: str, max_chars=6000) -> dict:
        import hashlib
        from google.genai import types
        huella = hashlib.md5(texto.encode()).hexdigest()[:10]
        for d in self.cache.values():           # reutiliza una respuesta válida de cualquier modelo
            if d["id"] == id_ and d["clave"].endswith(huella):
                return d
        prompt = (INSTRUCCIONES.format(esquema=ESQUEMAS[modalidad]) +
                  f"\nModalidad: {modalidad}\nTexto:\n<<<\n{texto[:max_chars]}\n>>>")
        campos, ok, intentos = {"error": "sin respuesta válida"}, False, 0
        while intentos < 6:
            intentos += 1
            try:
                r = self.cliente.models.generate_content(
                    model=self.modelo, contents=prompt,
                    config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"))
                campos, ok = json.loads(r.text), True
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                print(f"  {id_} [{self.modelo}]: {type(e).__name__} {msg[:120]}")
                if ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "404" in msg) and self._siguiente_modelo():
                    continue
                time.sleep(min(15 * intentos, 60))
        campos = validar_campos(campos, modalidad)
        d = {"clave": f"{self.modelo}|{id_}|{huella}", "id": id_, "modalidad": modalidad, "campos": campos,
             "modelo_llm": self.modelo, "fecha": time.strftime("%Y-%m-%d %H:%M"), "ok": ok}
        if ok:
            self.cache[d["clave"]] = d
            with open(self.cache_ruta, "a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        time.sleep(self.pausa)
        return d


def validar_campos(campos: dict, modalidad: str) -> dict:
    """Deja exactamente las claves pedidas (las que falten quedan en null) y fuerza la modalidad."""
    if isinstance(campos, list) and campos:
        campos = campos[0]
    if not isinstance(campos, dict):
        campos = {}
    salida = {k: campos.get(k) for k in CAMPOS[modalidad]}
    salida["modalidad"] = modalidad
    if modalidad == "audio" and not isinstance(salida.get("citas_clave"), list):
        salida["citas_clave"] = []
    return salida


def citas_verificables(campos: dict, texto: str) -> bool | None:
    """¿Cada cita clave aparece literalmente (normalizada) en la transcripción? Ayuda a auditar."""
    citas = campos.get("citas_clave") or []
    if not citas:
        return None
    t = normalizar(texto)
    return all(normalizar(c) in t for c in citas)


# ═══════════════════════════════════════════════ gráficos visibles en GitHub
def mostrar(fig, nombre: str, ancho=1000, alto=520):
    """Muestra el gráfico Plotly interactivo y además una copia PNG estática.

    GitHub no ejecuta JavaScript al mostrar notebooks: sin la PNG el gráfico no se vería.
    La PNG necesita `kaleido` (pip install kaleido; la primera vez descarga Chrome).
    """
    from IPython.display import Image, display
    FIGURAS.mkdir(exist_ok=True)
    fig.update_layout(width=ancho, height=alto, template="plotly_white")
    fig.write_html(FIGURAS / f"{nombre}.html", include_plotlyjs="cdn")
    try:
        fig.write_image(FIGURAS / f"{nombre}.png", scale=1.5)
        fig.show()
        display(Image(filename=str(FIGURAS / f"{nombre}.png")))
    except Exception as e:  # noqa: BLE001
        print(f"(sin PNG estática: {type(e).__name__}. Instala kaleido para que el gráfico se vea en GitHub)")
        fig.show()


def miniatura(origen: Path, destino: Path, lado=900, calidad=80) -> Path:
    from PIL import Image, ImageOps
    destino.parent.mkdir(parents=True, exist_ok=True)
    im = ImageOps.exif_transpose(Image.open(origen)).convert("RGB")
    im.thumbnail((lado, lado))
    im.save(destino, quality=calidad)
    return destino
