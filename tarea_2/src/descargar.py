"""Descarga y registra los dos corpus de la Tarea 2 (documentos e audio).

Se llama desde 01_extraccion.ipynb (o a mano: `python -m src.descargar` desde tarea_2/).
Todo lo descargado queda en datos/raw/, que NO se versiona en Git.

Corpus A · Documentos (25 = 8 impresos + 8 fotos + 9 manuscritos)
    1. Documentos peruanos ya verificados en la clase de PaddleOCR (Sesión 10): resolución de
       Mache, Mercurio Peruano, billetes de intis, placa de la calle Capón, Acta de la
       Independencia, tarjeta de Ricardo Palma, recibo de 1895, etc.
    2. Capturas propias opcionales en datos/propios/<tipo>/ (fotos del celular o manuscritos
       propios sin datos personales), registradas en datos/propios/propios.csv.
    3. Si faltan, se completan con búsquedas en la API de Wikimedia Commons, aceptando solo
       archivos con licencia libre (dominio público, CC0, CC BY, CC BY-SA) y con alguna
       categoría peruana. Siempre se bajan algunos de reserva por si hay que descartar.

Corpus B · Audio (>= 30 min, >= 20 clips, >= 1 multihablante)
    1. Entrevistas de Wikimedia Commons en español peruano (las de la Sesión 13 y otras que
       encuentre la búsqueda, con licencia libre).
    2. OpenSLR 73 (Google, CC BY-SA 4.0): español peruano leído con transcripción humana.
       Se arma una "sesión de lectura" por hablante uniendo 10 frases suyas (clips de ~50 s)
       y se guarda su transcripción oficial. Se leen solo los bytes necesarios (HTTP Range),
       como en datos.py de la Sesión 13.
    3. Una entrevista simulada con dos hablantes reales de OpenSLR 73 alternando turnos, con
       la verdad de quién habla cuándo (para diarización), como en datos.py de la Sesión 13.

Buenas prácticas: User-Agent descriptivo, 1-2 s de pausa entre solicitudes y registro de
fuente, autor y licencia de cada archivo.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]          # carpeta tarea_2/
RAW = RAIZ / "datos" / "raw"
DIR_DOCS = RAW / "documentos"
DIR_AUDIO = RAW / "audio"
PROPIOS = RAIZ / "datos" / "propios"

UA = {"User-Agent": "tarea2-ocr-asr-peru/1.1 (https://github.com/d2cml-ai/Data-Science-Python; curso Ciencia de Datos con Python UP 2026-II, uso academico) python-urllib"}
API = "https://commons.wikimedia.org/w/api.php"
SR = 16_000
PAUSA = 2.0   # segundos entre solicitudes a Commons

CUOTAS = {"impreso": 8, "foto": 8, "manuscrito": 9}
RESERVA = 3   # documentos extra por tipo, para reemplazar descartes

# ─────────────────────────────────────────── documentos verificados en la Sesión 10
# (archivo exacto en Commons, tipo, descripción corta). Licencias y autores se leen de la API.
# Miniaturas de 1920 px: Wikimedia solo sirve sin límite los anchos estándar (960, 1280, 1920...).
DOCS_CONOCIDOS = [
    ("Acta de Creación de los Símbolos para el Distrito de Mache.jpg", "impreso", "Resolución de alcaldía de Mache (La Libertad), 2016"),
    ("Pisco-punch-mercurio-peruano-1791.jpg", "impreso", "Mercurio Peruano, 1791 (imprenta colonial)"),
    ("10 Intis 1987(front).jpg", "impreso", "Billete de 10 intis, 1987"),
    ("100 Intis 1987(Front).jpg", "impreso", "Billete de 100 intis, 1987"),
    ("1000 Intis 1988(Front).jpg", "impreso", "Billete de 1000 intis, 1988"),
    ("10000 Intis 1988(Front).jpg", "impreso", "Billete de 10 000 intis, 1988"),
    ("500000 Intis 1988(Front).jpg", "impreso", "Billete de 500 000 intis, 1988"),
    ("5000000 Intis 1990(Front).jpg", "impreso", "Billete de 5 000 000 intis, 1990"),
    ("50 Intis 1987(Front).jpg", "impreso", "Billete de 50 intis, 1987 (reserva)"),
    ("Placa Calle Capon.jpg", "foto", "Placa de la calle Capón, Barrio Chino de Lima (foto en la calle)"),
    ("Cartel Picante de cuy.JPG", "foto", "Cartel «Picante de cuy», Ingenio, Junín (foto de rótulo)"),
    ("Carteles chicha.jpg", "foto", "Carteles chicha en Lima (foto en la calle)"),
    ("Acta de Independencia Peru.jpg", "manuscrito", "Acta de la Independencia del Perú, 1821"),
    ("Tarjeta de Ricardo Palma a Luis Antonio Eguiguren Escudero 24 octubre 1912.png", "manuscrito",
     "Tarjeta manuscrita de Ricardo Palma, 1912"),
    ("Peruvian Telephone Company receipt Lima 1895.JPG", "manuscrito", "Recibo de la Peruvian Telephone Company llenado a mano, 1895"),
    ("Menu Levaggi 09052010.jpg", "manuscrito", "Pizarra de menú escrita a mano con tiza, Lince, 2010"),
]

# Búsquedas para completar cuotas (CirrusSearch de Commons). Se filtran por licencia y categoría peruana.
BUSQUEDAS_DOCS = {
    "impreso": ["El Peruano diario portada", "Gaceta de Lima impreso", "periódico Lima siglo XIX",
                "Constitución Política del Perú impresa", "decreto supremo Perú documento", "libro impreso Lima 1800"],
    "foto": ["placa conmemorativa Lima", "placa Cusco", "letrero Lima calle", "cartel Arequipa",
             "placa Arequipa", "aviso Trujillo Perú", "placa Plaza de Armas Perú", "letrero Iquitos"],
    "manuscrito": ["carta manuscrita Perú", "manuscrito colonial Lima", "carta autógrafa Lima",
                   "Archivo General de la Nación Perú manuscrito", "partida de bautismo Perú",
                   "Nueva corónica y buen gobierno página", "carta de San Martín Lima", "carta de Bolívar Lima",
                   "manuscrito Perú siglo XIX", "carta manuscrita Cusco", "documento colonial Perú",
                   "libro parroquial Perú", "testamento Lima colonial", "carta Túpac Amaru",
                   "Archivo Regional Cusco", "manuscrito Arequipa", "partida de nacimiento Perú",
                   "carta autógrafa peruano", "manuscrito Trujillo Perú"],
}

# ─────────────────────────────────────────── audio
AUDIO_CONOCIDOS = [   # (página de Commons por número, idioma, descripción) — las de la Sesión 13.
    # Se usa el número de página (curid) y no el título, porque los títulos contienen nombres de personas.
    ("pageid:119251161", "es", "Entrevista en la calle a una vendedora de frutas, El Agustino (Lima), 2022"),
    ("pageid:119251160", "es", "Entrevista en la calle a dos vendedoras de dulces, El Agustino (Lima), 2022"),
    ("pageid:121219890", "es", "Dirigente de una organización de mujeres indígenas en conferencia de prensa, 2022"),
    ("buscar:Entrevista sobre Aurinsha Shansha Áncash", "qwh",
     "Entrevista en quechua de Huaylas sobre una planta medicinal (bonificación MMS)"),
]
BUSQUEDAS_AUDIO = ["entrevista Lima", "entrevista Perú", "entrevista Wikimedia Perú", "testimonio Perú",
                   "entrevista Cusco", "entrevista Ayacucho", "entrevista Arequipa", "conversatorio Lima",
                   "historia oral Perú", "español peruano"]

PERU = re.compile(r"per[uú]|lima|cusco|cuzco|arequipa|trujillo|ayacucho|puno|piura|iquitos|callao|jun[ií]n|"
                  r"[aá]ncash|cajamarca|huancayo|tacna|loreto|\bica\b|chiclayo|huaraz|moquegua|tumbes|"
                  r"apur[ií]mac|hu[aá]nuco|pasco|ucayali|amazonas \(per|la libertad|lambayeque|madre de dios|"
                  r"san mart[ií]n \(per|\bintis?\b|quechua|\bincas?\b", re.I)
LICENCIA_OK = re.compile(r"^(cc0|public domain|pd|cc[ -]by(-sa)?[ -]?\d|cc[ -]by(-sa)?$|attribution|"
                         r"cc-by-sa-|cc-by-|no restrictions)", re.I)
LICENCIA_NO = re.compile(r"\bnc\b|\bnd\b|non-?commercial|noderiv", re.I)


# ─────────────────────────────────────────── utilidades HTTP
def _get(url, rango=None, intentos=7):
    import urllib.error
    cab = dict(UA)
    if rango:
        cab["Range"] = f"bytes={rango[0]}-{rango[1]}"
    for k in range(intentos):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=cab), timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404 or k == intentos - 1:
                raise
            # 429 = demasiadas solicitudes: Wikimedia pide esperar (cabecera Retry-After)
            espera = int(e.headers.get("Retry-After") or 0) if e.code == 429 else 0
            espera = max(espera, 30 * (k + 1) if e.code == 429 else 5 * (k + 1))
            print(f"   HTTP {e.code}: espero {espera} s y reintento ({k + 1}/{intentos - 1})")
            time.sleep(espera)
        except Exception as e:  # cortes de red
            if k == intentos - 1:
                raise
            print(f"   {type(e).__name__}: reintento en {5 * (k + 1)} s")
            time.sleep(5 * (k + 1))


def _api(**params):
    params = {"format": "json", "formatversion": "2", **params}
    datos = json.loads(_get(API + "?" + urllib.parse.urlencode(params)))
    time.sleep(PAUSA)
    return datos


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _limpiar_html(x):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(x or ""))).strip()


def _info_paginas(paginas: list[dict], ancho=1920) -> list[dict]:
    """Normaliza la respuesta de la API: url, mime, licencia, autor, categorías y duración."""
    filas = []
    for p in paginas:
        if p.get("missing") or not p.get("imageinfo"):
            continue
        ii = p["imageinfo"][0]
        meta = ii.get("extmetadata", {})
        dur = None
        for m in ii.get("metadata") or []:
            if m.get("name") in ("length", "playtime_seconds", "duration"):
                try:
                    dur = float(m["value"])
                except (TypeError, ValueError):
                    pass
        cats = " | ".join(c["title"].replace("Category:", "") for c in p.get("categories", []))
        filas.append({
            "titulo": p["title"].replace("File:", ""),
            "url_descarga": ii.get("thumburl") if ii.get("thumburl") and ancho else ii.get("url"),
            "url_original": ii.get("url"),
            "fuente_url": ii.get("descriptionurl"),
            "mime": ii.get("mime", ""),
            "ancho": ii.get("width"), "alto": ii.get("height"),
            "bytes": ii.get("size"),
            "duracion_meta": dur,
            "licencia": _limpiar_html(meta.get("LicenseShortName", {}).get("value", "")),
            "autor": _limpiar_html(meta.get("Artist", {}).get("value", ""))[:200],
            "fecha_obra": _limpiar_html(meta.get("DateTimeOriginal", {}).get("value", ""))[:60],
            "categorias": cats,
        })
    return filas


def info_commons(titulos: list[str], ancho=1920) -> pd.DataFrame:
    """Metadatos de archivos concretos de Commons (de a 20 por consulta)."""
    filas = []
    for i in range(0, len(titulos), 20):
        lote = titulos[i:i + 20]
        extra = {"iiurlwidth": ancho} if ancho else {}
        r = _api(action="query", titles="|".join("File:" + t for t in lote), prop="imageinfo|categories",
                 iiprop="url|size|mime|extmetadata|metadata", cllimit="max", redirects=1, **extra)
        q = r.get("query", {})
        # la API normaliza títulos (guiones bajos, mayúscula inicial) y sigue redirecciones:
        # se guarda el título pedido para poder emparejar cada respuesta con su solicitud
        hacia_atras = {}
        for paso in q.get("normalized", []) + q.get("redirects", []):
            hacia_atras[paso["to"]] = hacia_atras.get(paso["from"], paso["from"])
        for f in _info_paginas(q.get("pages", []), ancho):
            pedido = hacia_atras.get("File:" + f["titulo"], "File:" + f["titulo"]).replace("File:", "")
            filas.append(f | {"pedido": pedido.replace("_", " ")})
    return pd.DataFrame(filas)


def buscar_commons(consulta: str, tipo_archivo="bitmap", n=25, ancho=1920) -> pd.DataFrame:
    extra = {"iiurlwidth": ancho} if ancho else {}
    r = _api(action="query", generator="search", gsrsearch=f"{consulta} filetype:{tipo_archivo}",
             gsrnamespace=6, gsrlimit=n, prop="imageinfo|categories",
             iiprop="url|size|mime|extmetadata|metadata", cllimit="max", **extra)
    paginas = sorted(r.get("query", {}).get("pages", []), key=lambda p: p.get("index", 0))
    return pd.DataFrame(_info_paginas(paginas, ancho))


def url_curid(titulo: str) -> str:
    """URL estable de Commons por número de página (?curid=), sin el título del archivo.

    Se usa para las entrevistas: el nombre del archivo suele traer el nombre de la persona.
    """
    try:
        r = _api(action="query", titles="File:" + titulo, redirects=1)
        pagina = r["query"]["pages"][0]
        return f"https://commons.wikimedia.org/?curid={pagina['pageid']}"
    except Exception:  # noqa: BLE001
        return "https://commons.wikimedia.org (ver título seudonimizado)"


def licencia_valida(lic: str) -> bool:
    lic = (lic or "").strip()
    return bool(lic) and bool(LICENCIA_OK.search(lic)) and not LICENCIA_NO.search(lic)


def es_peruano(fila) -> bool:
    return bool(PERU.search(f"{fila['titulo']} {fila['categorias']}"))


# ─────────────────────────────────────────── corpus A: documentos
def _cargar_inventario(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta, dtype=str, keep_default_na=False) if ruta.exists() else pd.DataFrame()


def documentos(buscar=True, verbose=True) -> pd.DataFrame:
    """Baja los documentos y devuelve el inventario (datos/raw/documentos/inventario.csv)."""
    DIR_DOCS.mkdir(parents=True, exist_ok=True)
    ruta_inv = DIR_DOCS / "inventario.csv"
    inv = _cargar_inventario(ruta_inv)
    vistos = set(inv["titulo"]) if len(inv) else set()
    filas = inv.to_dict("records") if len(inv) else []

    def agregar(info: dict, tipo: str, descripcion: str, origen: str):
        n = len(filas) + 1
        ext = Path(urllib.parse.urlparse(info["url_descarga"]).path).suffix.lower() or ".jpg"
        ext = ".jpg" if ext in (".jpeg", ".tif", ".tiff") and "thumb" in info["url_descarga"] else ext
        destino = DIR_DOCS / f"doc_{n:03d}{ext}"
        if not destino.exists():
            destino.write_bytes(_get(info["url_descarga"]))
            time.sleep(PAUSA)
        filas.append({"id": f"doc_{n:03d}", "archivo": destino.relative_to(RAIZ).as_posix(), "tipo": tipo,
                      "titulo": info["titulo"], "descripcion": descripcion, "origen": origen,
                      "fuente_url": info["fuente_url"], "licencia": info["licencia"], "autor": info["autor"],
                      "fecha_obra": info.get("fecha_obra", ""), "fecha_descarga": time.strftime("%Y-%m-%d"),
                      "sha256": sha256(destino)})
        vistos.add(info["titulo"])
        pd.DataFrame(filas).to_csv(ruta_inv, index=False, encoding="utf-8")   # guardar avance
        if verbose:
            print(f"  + doc_{n:03d} [{tipo:<10}] {info['titulo'][:70]}  ({info['licencia']})")

    # 1) los documentos ya verificados en la Sesión 10
    faltan = [d for d in DOCS_CONOCIDOS if d[0] not in vistos]
    if faltan:
        print(f"Documentos de la Sesión 10: {len(faltan)} por bajar")
        meta = info_commons([d[0] for d in faltan])
        meta = meta.drop_duplicates("pedido").set_index("pedido") if len(meta) else meta
        for titulo, tipo, desc in faltan:
            clave = titulo.replace("_", " ")
            if clave not in meta.index:
                print(f"  ! no encontrado en Commons: {titulo}")
                continue
            info = meta.loc[clave].to_dict()
            agregar(info, tipo, desc, "Wikimedia Commons (verificado en Sesión 10)")

    # 2) capturas propias: datos/propios/propios.csv con columnas archivo,tipo,descripcion,publicable
    reg = PROPIOS / "propios.csv"
    if reg.exists():
        propios = pd.read_csv(reg, dtype=str, keep_default_na=False)
        for p in propios.itertuples():
            titulo = f"propio:{p.archivo}"
            if titulo in vistos or not (PROPIOS / p.archivo).exists():
                continue
            n = len(filas) + 1
            src = PROPIOS / p.archivo
            destino = DIR_DOCS / f"doc_{n:03d}{src.suffix.lower()}"
            destino.write_bytes(src.read_bytes())
            filas.append({"id": f"doc_{n:03d}", "archivo": destino.relative_to(RAIZ).as_posix(), "tipo": p.tipo,
                          "titulo": titulo, "descripcion": p.descripcion, "origen": "Captura propia",
                          "fuente_url": "", "licencia": "CC BY 4.0 (captura propia del estudiante)",
                          "autor": "Estudiante (autor de la captura)", "fecha_obra": "",
                          "fecha_descarga": time.strftime("%Y-%m-%d"), "sha256": sha256(destino),
                          "publicable": getattr(p, "publicable", "no")})
            vistos.add(titulo)
            print(f"  + doc_{n:03d} [{p.tipo:<10}] captura propia {p.archivo}")

    # 3) completar cuotas (+ reserva) con búsquedas en Commons
    if buscar:
        tabla = pd.DataFrame(filas)
        for tipo, consultas in BUSQUEDAS_DOCS.items():
            meta_tipo = CUOTAS[tipo] + RESERVA
            for q in consultas:
                tiene = int((tabla["tipo"] == tipo).sum()) if len(tabla) else 0
                if tiene >= meta_tipo:
                    break
                try:
                    cand = buscar_commons(q, "bitmap")
                except Exception as e:
                    print(f"  ! búsqueda «{q}» falló: {e}")
                    continue
                for c in cand.to_dict("records"):
                    if (tabla["tipo"] == tipo).sum() >= meta_tipo:
                        break
                    ok = (c["titulo"] not in vistos and licencia_valida(c["licencia"]) and es_peruano(c)
                          and c["mime"] in ("image/jpeg", "image/png", "image/tiff")
                          and (c["ancho"] or 0) >= 600)
                    if ok:
                        agregar(c, tipo, f"Búsqueda Commons: «{q}»", "Wikimedia Commons (búsqueda API)")
                        tabla = pd.DataFrame(filas)

    inv = pd.DataFrame(filas)
    inv.to_csv(ruta_inv, index=False, encoding="utf-8")
    if verbose:
        print("\nDocumentos por tipo:", inv["tipo"].value_counts().to_dict())
    return inv


# ─────────────────────────────────────────── corpus B: audio
class ZipRemoto(io.RawIOBase):
    """Archivo remoto leído por partes con HTTP Range (tomado de datos.py, Sesión 13)."""
    BLOQUE = 1 << 20

    def __init__(self, url):
        self.url, self.pos, self.buf_ini, self.buf = url, 0, 0, b""
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        fin = self.pos + n
        if not (self.buf_ini <= self.pos and fin <= self.buf_ini + len(self.buf)):
            self.buf_ini = self.pos
            self.buf = _get(self.url, (self.pos, min(self.pos + max(n, self.BLOQUE), self.size) - 1))
        datos = self.buf[self.pos - self.buf_ini:fin - self.buf_ini]
        self.pos += len(datos)
        return datos


def _wav_16k(contenido: bytes) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(io.BytesIO(contenido), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype("float32")
    return x


def _duracion(ruta: Path) -> float:
    import soundfile as sf
    try:
        return float(sf.info(str(ruta)).duration)
    except Exception:
        from faster_whisper import decode_audio
        return len(decode_audio(str(ruta), sampling_rate=SR)) / SR


OPENSLR73 = "https://www.openslr.org/resources/73/es_pe_{sexo}.zip"


def openslr73(minutos_objetivo: float, frases_por_sesion=10, semilla=2026, verbose=True):
    """Sesiones de lectura de OpenSLR 73: un clip por hablante, uniendo `frases_por_sesion` frases.

    Devuelve (sesiones, entrevista_simulada) como listas de dicts. Para cuando la suma de
    duraciones alcanza `minutos_objetivo`. Las transcripciones oficiales se guardan como
    referencia candidata (hechas por personas para el corpus; hay que verificarlas al oído).
    """
    import soundfile as sf
    destino = DIR_AUDIO / "openslr73"
    destino.mkdir(parents=True, exist_ok=True)
    zips, tablas = {}, {}
    for sexo in ["female", "male"]:
        zips[sexo] = zipfile.ZipFile(ZipRemoto(OPENSLR73.format(sexo=sexo)))
        lineas = zips[sexo].read("line_index.tsv").decode("utf-8").strip().splitlines()
        t = pd.DataFrame([l.split("\t")[:2] for l in lineas], columns=["id", "texto"])
        t["hablante"] = t["id"].str.rsplit("_", n=1).str[0]
        t = t[t["texto"].str.split().str.len().between(5, 25)]
        tablas[sexo] = t
    rng = random.Random(semilla)
    orden = {s: sorted(tablas[s]["hablante"].unique()) for s in tablas}
    for s in orden:
        rng.shuffle(orden[s])
    sesiones, total, k = [], 0.0, 0
    silencio = np.zeros(int(0.6 * SR), dtype=np.float32)
    # las dos primeras hablantes de cada lista se reservan para la entrevista simulada
    reserva = {s: orden[s][0] for s in orden}
    cola = [h for par in zip(orden["female"][1:], orden["male"][1:]) for h in par]
    while total < minutos_objetivo * 60 and k < len(cola):
        h = cola[k]
        k += 1
        sexo = "female" if h in set(orden["female"]) else "male"
        frases = tablas[sexo][tablas[sexo]["hablante"] == h].head(frases_por_sesion)
        if len(frases) < frases_por_sesion:
            continue
        sid = f"openslr73_{h}"
        ruta = destino / f"{sid}.wav"
        meta = destino / f"{sid}.json"
        if ruta.exists() and meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
        else:
            partes = []
            for f in frases.itertuples():
                partes += [_wav_16k(zips[sexo].read(f"{f.id}.wav")), silencio]
            sf.write(ruta, np.concatenate(partes), SR)
            m = {"sesion": sid, "hablante": h, "sexo": "mujer" if sexo == "female" else "hombre",
                 "frases": list(frases["id"]), "referencia": " ".join(t.strip() for t in frases["texto"])}
            meta.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        dur = _duracion(ruta)
        total += dur
        sesiones.append(m | {"ruta": ruta, "duracion": dur})
        if verbose:
            print(f"  + {sid} ({m['sexo']}, {dur:.0f} s) · acumulado {total / 60:.1f} min")

    # entrevista simulada: 7 turnos de una mujer y 7 de un hombre, alternados
    wav, verdad = destino / "entrevista_simulada.wav", destino / "entrevista_simulada.csv"
    if not (wav.exists() and verdad.exists()):
        a = tablas["female"][tablas["female"]["hablante"] == reserva["female"]].head(7)
        b = tablas["male"][tablas["male"]["hablante"] == reserva["male"]].head(7)
        audio, filas, t = [], [], 0.0
        pausa = np.zeros(int(0.7 * SR), dtype=np.float32)
        for fa, fb in zip(a.itertuples(), b.itertuples()):
            for f, sexo, quien in [(fa, "female", "Entrevistadora"), (fb, "male", "Entrevistado")]:
                x = _wav_16k(zips[sexo].read(f"{f.id}.wav"))
                filas.append({"inicio": round(t, 2), "fin": round(t + len(x) / SR, 2), "hablante": quien, "texto": f.texto.strip()})
                audio += [x, pausa]
                t += (len(x) + len(pausa)) / SR
        sf.write(wav, np.concatenate(audio), SR)
        pd.DataFrame(filas).to_csv(verdad, index=False, encoding="utf-8")
    sim = pd.read_csv(verdad)
    simulada = {"sesion": "entrevista_simulada", "ruta": wav, "duracion": _duracion(wav),
                "referencia": " ".join(sim["texto"]), "verdad": verdad,
                "hablantes": [reserva["female"], reserva["male"]]}
    return sesiones, simulada


def audios(minutos_objetivo=36, buscar=True, max_min_commons=18, verbose=True) -> pd.DataFrame:
    """Baja el corpus de audio y devuelve el inventario (datos/raw/audio/inventario.csv).

    Una fila por archivo fuente (las entrevistas largas se cortan en clips en el notebook).
    """
    DIR_AUDIO.mkdir(parents=True, exist_ok=True)
    dir_c = DIR_AUDIO / "commons"
    dir_c.mkdir(exist_ok=True)
    ruta_inv = DIR_AUDIO / "inventario.csv"
    if ruta_inv.exists():
        inv = pd.read_csv(ruta_inv, dtype={"id_fuente": str}, keep_default_na=False)
        if verbose:
            print(f"Inventario de audio ya existe ({len(inv)} fuentes). Bórralo para volver a bajar.")
        return inv
    filas = []

    def agregar_commons(info, idioma, desc, origen):
        n = sum(f["origen"].startswith("Wikimedia") for f in filas) + 1
        ext = Path(urllib.parse.urlparse(info["url_original"]).path).suffix.lower()
        ruta = dir_c / f"commons_{n:02d}{ext}"
        if not ruta.exists():
            ruta.write_bytes(_get(info["url_original"]))
            time.sleep(PAUSA)
        dur = _duracion(ruta)
        filas.append({"id_fuente": f"commons_{n:02d}", "archivo": ruta.relative_to(RAIZ).as_posix(), "idioma": idioma,
                      "titulo": info["titulo"], "descripcion": desc, "origen": origen,
                      "fuente_url": info["fuente_url"], "licencia": info["licencia"], "autor": info["autor"],
                      "duracion_seg": round(dur, 1), "fecha_descarga": time.strftime("%Y-%m-%d"),
                      "multihablante": "posible", "sha256": sha256(ruta), "referencia_oficial": ""})
        if verbose:
            print(f"  + commons_{n:02d} [{idioma}] {info['titulo'][:65]} ({dur:.0f} s, {info['licencia']})")

    print("Entrevistas de Wikimedia Commons (Sesión 13)")
    for clave, idioma, desc in AUDIO_CONOCIDOS:
        try:
            if clave.startswith("pageid:"):
                r = _api(action="query", pageids=clave.split(":", 1)[1], prop="imageinfo|categories",
                         iiprop="url|size|mime|extmetadata|metadata", cllimit="max")
                info = _info_paginas(r["query"]["pages"], 0)[0]
            else:
                info = buscar_commons(clave.split(":", 1)[1], "audio", n=5, ancho=0).iloc[0].to_dict()
            agregar_commons(info, idioma, desc, "Wikimedia Commons (verificado en Sesión 13)")
        except Exception as e:  # noqa: BLE001
            print(f"  ! no encontrado: {clave} ({e})")

    if buscar:
        print("Búsqueda de más entrevistas en español peruano en Commons")
        vistos = {f["titulo"] for f in filas}
        for q in BUSQUEDAS_AUDIO:
            min_es = sum(f["duracion_seg"] for f in filas if f["idioma"] == "es") / 60
            if min_es >= max_min_commons:
                break
            try:
                cand = buscar_commons(q, "audio", n=30, ancho=0)
            except Exception as e:
                print(f"  ! búsqueda «{q}» falló: {e}")
                continue
            for c in cand.to_dict("records"):
                d = c["duracion_meta"]
                ok = (c["titulo"] not in vistos and licencia_valida(c["licencia"]) and es_peruano(c)
                      and not re.search(r"quechua|aimara|aymara|english|inglés|music|música|song|canción|himno",
                                        f"{c['titulo']} {c['categorias']}", re.I)
                      and (d is None or 20 <= d <= 900) and (c["bytes"] or 0) < 60e6)
                if ok:
                    agregar_commons(c, "es", f"Búsqueda Commons: «{q}» (verificar idioma y contenido)",
                                    "Wikimedia Commons (búsqueda API)")
                    vistos.add(c["titulo"])
                    if sum(f["duracion_seg"] for f in filas if f["idioma"] == "es") / 60 >= max_min_commons:
                        break

    min_es = sum(f["duracion_seg"] for f in filas if f["idioma"] == "es") / 60
    print(f"\nCommons en español: {min_es:.1f} min. OpenSLR 73 completa hasta {minutos_objetivo} min.")
    sesiones, sim = openslr73(max(minutos_objetivo - min_es, 12))
    for s in sesiones:
        filas.append({"id_fuente": s["sesion"], "archivo": Path(s["ruta"]).relative_to(RAIZ).as_posix(), "idioma": "es",
                      "titulo": f"OpenSLR 73 · hablante {s['hablante']} ({s['sexo']}), 10 frases leídas",
                      "descripcion": "Sesión de lectura: frases leídas por un mismo hablante, unidas con 0,6 s de pausa",
                      "origen": "OpenSLR 73", "fuente_url": "https://www.openslr.org/73/",
                      "licencia": "CC BY-SA 4.0", "autor": "Google (Guevara-Rukoz et al., 2020)",
                      "duracion_seg": round(s["duracion"], 1), "fecha_descarga": time.strftime("%Y-%m-%d"),
                      "multihablante": "no", "sha256": sha256(Path(s["ruta"])), "referencia_oficial": s["referencia"]})
    filas.append({"id_fuente": "entrevista_simulada", "archivo": Path(sim["ruta"]).relative_to(RAIZ).as_posix(), "idioma": "es",
                  "titulo": "Entrevista simulada: 2 hablantes de OpenSLR 73 alternando 14 turnos",
                  "descripcion": "Turnos reales de dos hablantes unidos; se conoce quién habla cuándo (diarización)",
                  "origen": "OpenSLR 73", "fuente_url": "https://www.openslr.org/73/", "licencia": "CC BY-SA 4.0",
                  "autor": "Google (Guevara-Rukoz et al., 2020); montaje como en datos.py de la Sesión 13",
                  "duracion_seg": round(sim["duracion"], 1), "fecha_descarga": time.strftime("%Y-%m-%d"),
                  "multihablante": "si", "sha256": sha256(Path(sim["ruta"])), "referencia_oficial": sim["referencia"]})
    inv = pd.DataFrame(filas)
    inv.to_csv(ruta_inv, index=False, encoding="utf-8")
    es = inv[inv["idioma"] == "es"]
    print(f"Total en español: {es['duracion_seg'].sum() / 60:.1f} min en {len(es)} archivos fuente")
    return inv


if __name__ == "__main__":
    print(documentos()[["id", "tipo", "titulo", "licencia"]].to_string(index=False))
    print(audios()[["id_fuente", "idioma", "duracion_seg", "licencia"]].to_string(index=False))
