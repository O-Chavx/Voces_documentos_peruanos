"""Dashboard de la Tarea 2 · Voces y documentos peruanos: del archivo escaneado al texto buscable.

Solo lee los CSV que generan los notebooks (no carga modelos de OCR, ASR ni LLM):

    streamlit run app.py
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
TIPOS = ["impreso", "foto", "manuscrito"]

st.set_page_config(page_title="Voces y documentos peruanos", page_icon="📜", layout="wide")


# ───────────────────────────────────────────── carga de datos
@st.cache_data
def leer_csv(nombre: str) -> pd.DataFrame:
    ruta = DATOS / nombre
    if not ruta.exists():
        return pd.DataFrame()
    return pd.read_csv(ruta, keep_default_na=False, dtype={"texto_referencia": str})


@st.cache_data
def leer_json(nombre: str) -> dict:
    ruta = DATOS / nombre
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


docs = leer_csv("documentos.csv")
aud = leer_csv("transcripciones.csv")
llm = leer_csv("extraccion_llm.csv")
ev_doc = leer_csv("eval_documentos.csv")
ev_aud = leer_csv("eval_audio.csv")
metricas = leer_csv("metricas_modelos.csv")
auditoria = leer_csv("auditoria_llm.csv")
fuentes = leer_csv("fuentes.csv")
resumen = leer_json("resumen_resultados.json")
for c in ["valor", "promedio_por_archivo", "segundos_por_item", "rtf"]:
    if c in metricas:
        metricas[c] = pd.to_numeric(metricas[c], errors="coerce")
for df, cols in [(docs, ["confianza_mediana", "pct_lineas_dudosas", "n_lineas"]), (aud, ["duracion_seg", "rtf", "inicio_seg", "fin_seg"])]:
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

if docs.empty or aud.empty:
    st.error("No encuentro datos/documentos.csv o datos/transcripciones.csv. Ejecuta primero 01_extraccion.ipynb.")
    st.stop()

campos_llm = {r.id: json.loads(r.campos_json) for r in llm.itertuples()} if not llm.empty else {}


def sin_tildes(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(t).lower()) if unicodedata.category(c) != "Mn")


def contiene(serie: pd.Series, consulta: str) -> pd.Series:
    if not consulta:
        return pd.Series(True, index=serie.index)
    q = sin_tildes(consulta)
    return serie.map(lambda t: q in sin_tildes(t))


def resaltar(texto: str, consulta: str) -> str:
    """Marca en amarillo las apariciones de la búsqueda (sin distinguir tildes ni mayúsculas)."""
    texto = str(texto).replace("<", "&lt;").replace(">", "&gt;")
    if not consulta:
        return texto
    base, q = sin_tildes(texto), sin_tildes(consulta)
    partes, i = [], 0
    for m in re.finditer(re.escape(q), base):
        partes += [texto[i:m.start()], f"<mark>{texto[m.start():m.end()]}</mark>"]
        i = m.end()
    return "".join(partes) + texto[i:]


def caja_texto(texto: str, consulta: str, alto=320):
    html = resaltar(texto, consulta).replace("\n", "<br>")
    html = re.sub(r"\[(HABLANTE_\d+)\]", r"<br><b>[\1]</b>", html).removeprefix("<br>")
    st.markdown(f"<div style='max-height:{alto}px;overflow-y:auto;padding:0.8rem;border:1px solid #8884;"
                f"border-radius:8px;font-family:ui-monospace,monospace;font-size:0.85rem;line-height:1.5'>{html}</div>",
                unsafe_allow_html=True)


def mostrar_campos(id_):
    c = campos_llm.get(id_)
    if not c:
        st.caption("Sin extracción del LLM para este registro.")
        return
    filas = [(k, ", ".join(v) if isinstance(v, list) else ("—" if v is None else str(v))) for k, v in c.items()]
    st.dataframe(pd.DataFrame(filas, columns=["campo", "valor"]), hide_index=True, width="stretch")
    if not llm.empty:
        st.caption(f"Modelo: {llm.set_index('id').at[id_, 'modelo_llm']} · null = el dato no aparece en el texto")


# ───────────────────────────────────────────── encabezado
st.title("¿Pueden los documentos escaneados y las entrevistas en audio convertirse en texto buscable?")
st.markdown(
    f"Pipeline de IA aplicado a **{len(docs)} documentos peruanos** (impresos, fotos de celular y manuscritos) "
    f"y **{aud['duracion_seg'].sum() / 60:.0f} minutos de audio** en español peruano: OCR con **PaddleOCR**, "
    "transcripción con **faster-whisper** (VAD + diarización), **seudonimización**, extracción de campos con un "
    "**LLM (Gemini)** y evaluación con **CER/WER** sobre 10 documentos y 10 clips transcritos a mano. "
    "Todo se calculó antes en los notebooks; esta app solo explora los resultados."
)

# ───────────────────────────────────────────── filtros
with st.sidebar:
    st.header("Filtros")
    consulta = st.text_input("🔎 Buscar en los textos OCR y ASR", placeholder="p. ej. Lima, independencia, intis")
    tipos_sel = st.multiselect("Tipo de documento", TIPOS, default=TIPOS)
    modelos_ocr = ["Corpus completo (PaddleOCR base)"] + (list(ev_doc["modelo"].unique()) if not ev_doc.empty else [])
    modelo_ocr = st.selectbox("Modelo OCR", modelos_ocr,
                              help="Los modelos alternativos solo se ejecutaron sobre los 10 documentos de evaluación.")
    modelos_asr = ["Corpus completo (faster-whisper small)"] + (list(ev_aud["modelo"].unique()) if not ev_aud.empty else [])
    modelo_asr = st.selectbox("Modelo ASR", modelos_asr,
                              help="Los modelos alternativos solo se ejecutaron sobre los 10 clips de evaluación.")
    condiciones = sorted(aud["condicion"].unique())
    cond_sel = st.multiselect("Fuente / condición del audio", condiciones, default=condiciones)
    dmin, dmax = float(aud["duracion_seg"].min()), float(aud["duracion_seg"].max())
    dur_sel = st.slider("Duración del clip (s)", min_value=0.0, max_value=max(dmax, 1.0), value=(0.0, max(dmax, 1.0)), step=5.0)
    st.caption("Los filtros actualizan los indicadores de búsqueda, los gráficos y las tablas.")

# documentos visibles según filtros (texto del modelo elegido)
d_vis = docs[docs["tipo"].isin(tipos_sel)].copy()
d_vis["texto"] = d_vis["texto_ocr"]
if modelo_ocr in set(ev_doc.get("modelo", [])):
    hip = ev_doc[ev_doc["modelo"] == modelo_ocr].set_index("id")["texto_hipotesis"]
    d_vis = d_vis[d_vis["id"].isin(hip.index)]
    d_vis["texto"] = d_vis["id"].map(hip)
d_vis = d_vis[contiene(d_vis["texto"], consulta)]

a_vis = aud[aud["condicion"].isin(cond_sel) & aud["duracion_seg"].between(*dur_sel)].copy()
a_vis["texto"] = a_vis["texto_asr"]
if modelo_asr in set(ev_aud.get("modelo", [])):
    hip = ev_aud[ev_aud["modelo"] == modelo_asr].set_index("id")["texto_hipotesis"]
    a_vis = a_vis[a_vis["id"].isin(hip.index)]
    a_vis["texto"] = a_vis["id"].map(hip)
a_vis = a_vis[contiene(a_vis["texto"], consulta)]

# ───────────────────────────────────────────── indicadores
m = st.columns(6)
m[0].metric("Documentos procesados", len(docs), help="8 impresos, 8 fotos y 9 manuscritos")
m[1].metric("Minutos de audio transcritos", f"{aud['duracion_seg'].sum() / 60:.1f}", help=f"{len(aud)} clips")
if resumen:
    m[2].metric(f"CER · {resumen['modelo_ocr_elegido']}", f"{resumen['cer']:.1%}",
                delta=f"{resumen['cer'] - resumen['cer_base']:+.1%} vs. base", delta_color="inverse")
    m[3].metric(f"WER · {resumen['modelo_asr_elegido']}", f"{resumen['wer']:.1%}",
                delta=f"{resumen['wer'] - resumen['wer_base']:+.1%} vs. base", delta_color="inverse")
    m[4].metric("Acuerdo LLM vs. auditoría humana", f"{resumen['pct_acuerdo']:.0%}", help=f"{resumen['acuerdo_llm_humano']} extracciones completamente correctas")
else:
    m[2].metric("CER", "—"); m[3].metric("WER", "—"); m[4].metric("Acuerdo LLM-humano", "—")
m[5].metric("Coincidencias de la búsqueda", f"{len(d_vis)} docs · {len(a_vis)} clips" if consulta else "—")

tab_res, tab_doc, tab_aud, tab_eval, tab_datos = st.tabs(
    ["📊 Resumen", "📄 Documentos", "🎙️ Audios", "🧪 Evaluación de modelos", "🗂️ Datos y fuentes"])

# ───────────────────────────────────────────── resumen
with tab_res:
    c1, c2 = st.columns(2)
    with c1:
        g = d_vis.groupby("tipo").agg(documentos=("id", "count"), confianza=("confianza_mediana", "median")).reindex(TIPOS).dropna(how="all").reset_index()
        fig = px.bar(g, x="tipo", y="documentos", color="confianza", color_continuous_scale="RdYlGn", range_color=(0.4, 1),
                     text="documentos", labels={"confianza": "confianza OCR mediana"},
                     title="Documentos por tipo (color: confianza mediana del OCR)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.histogram(a_vis, x="duracion_seg", color="condicion", nbins=25,
                           labels={"duracion_seg": "duración del clip (s)", "condicion": "condición"},
                           title="Duración de los clips por fuente")
        st.plotly_chart(fig, width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        d_vis["caracteres"] = d_vis["texto"].str.len()
        fig = px.box(d_vis, x="tipo", y="caracteres", points="all", category_orders={"tipo": TIPOS},
                     hover_data=["id", "titulo"], title="Longitud del texto OCR por tipo de documento")
        st.plotly_chart(fig, width="stretch")
    with c4:
        a_vis["palabras"] = a_vis["texto"].str.split().str.len()
        fig = px.scatter(a_vis, x="duracion_seg", y="palabras", color="condicion", hover_data=["id", "n_hablantes"],
                         labels={"duracion_seg": "duración (s)", "palabras": "palabras transcritas"},
                         title="Palabras transcritas vs. duración del clip")
        st.plotly_chart(fig, width="stretch")

# ───────────────────────────────────────────── explorador de documentos
with tab_doc:
    if d_vis.empty:
        st.info("Ningún documento cumple los filtros.")
    else:
        etiqueta = {r.id: f"{r.id} · {r.tipo} · {r.titulo[:60]}" for r in d_vis.itertuples()}
        sel = st.selectbox("Elige un documento", list(etiqueta), format_func=etiqueta.get)
        r = docs.set_index("id").loc[sel]
        col_img, col_txt = st.columns([2, 3])
        with col_img:
            if r.imagen_ejemplo and (RAIZ / r.imagen_ejemplo).exists():
                st.image(str(RAIZ / r.imagen_ejemplo), width="stretch")
            else:
                st.info("La imagen original no se publica (licencia o datos personales); solo su texto y metadatos.")
            st.caption(f"**{r.titulo}** — {r.descripcion}  \n{r.autor} · {r.licencia}")
            if r.fuente_url:
                st.markdown(f"[Ver ficha y licencia en la fuente]({r.fuente_url})")
            st.caption(f"{int(r.n_lineas)} líneas · confianza mediana {float(r.confianza_mediana or 0):.2f} · "
                       f"{float(r.pct_lineas_dudosas or 0):.0f}% de líneas con confianza < 0,90")
        with col_txt:
            texto = d_vis.set_index("id").at[sel, "texto"]
            st.subheader(f"Texto OCR · {modelo_ocr}")
            caja_texto(texto, consulta)
            st.subheader("Campos extraídos por el LLM")
            mostrar_campos(sel)
            if r.texto_referencia:
                st.subheader("Referencia manual y CER")
                if not ev_doc.empty and sel in set(ev_doc["id"]):
                    t = ev_doc[ev_doc["id"] == sel][["modelo", "cer", "segundos"]].rename(columns={"cer": "CER", "segundos": "segundos"})
                    st.dataframe(t.style.format({"CER": "{:.1%}", "segundos": "{:.1f}"}), hide_index=True, width="stretch")
                with st.expander("Ver transcripción de referencia (hecha a mano)"):
                    caja_texto(r.texto_referencia, consulta, alto=260)

# ───────────────────────────────────────────── explorador de audios
with tab_aud:
    if a_vis.empty:
        st.info("Ningún clip cumple los filtros.")
    else:
        etiqueta = {r.id: f"{r.id} · {r.condicion} · {r.duracion_seg:.0f} s · {r.n_hablantes} hablante(s)" for r in a_vis.itertuples()}
        sel = st.selectbox("Elige un clip", list(etiqueta), format_func=etiqueta.get)
        r = aud.set_index("id").loc[sel]
        c1, c2 = st.columns([2, 3])
        with c1:
            if r.audio_ejemplo and (RAIZ / r.audio_ejemplo).exists():
                st.audio(str(RAIZ / r.audio_ejemplo))
                st.caption("Clip publicable (OpenSLR 73, CC BY-SA 4.0).")
            else:
                st.info("Este audio no se redistribuye aquí (entrevista con personas reales). "
                        "Se muestra su transcripción seudonimizada y el enlace a la fuente con su licencia.")
            st.caption(f"**{r.titulo}**  \n{r.autor} · {r.licencia}  \nTramo {float(r.inicio_seg):.0f}–{float(r.fin_seg):.0f} s del original")
            if r.fuente_url:
                st.markdown(f"[Fuente y licencia]({r.fuente_url})")
            st.caption(f"Diarización: {r.diarizado} · RTF faster-whisper small: {float(r.rtf):.2f}")
            if r.alucinaciones:
                st.warning(f"Posible alucinación: {r.alucinaciones}")
        with c2:
            st.subheader(f"Transcripción · {modelo_asr}")
            caja_texto(a_vis.set_index("id").at[sel, "texto"], consulta)
            st.subheader("Campos extraídos por el LLM")
            mostrar_campos(sel)
        if not ev_aud.empty and sel in set(ev_aud["id"]):
            st.subheader("Comparación de modelos en este clip (referencia manual)")
            t = ev_aud[ev_aud["id"] == sel]
            st.dataframe(t[["modelo", "wer", "rtf"]].rename(columns={"wer": "WER", "rtf": "RTF"})
                         .style.format({"WER": "{:.1%}", "RTF": "{:.2f}"}), hide_index=True, width="stretch")
            cols = st.columns(len(t) + 1)
            with cols[0]:
                st.markdown("**Referencia manual**")
                caja_texto(r.texto_referencia, consulta, alto=260)
            for col, fila in zip(cols[1:], t.itertuples()):
                with col:
                    st.markdown(f"**{fila.modelo}** · WER {fila.wer:.0%}")
                    caja_texto(fila.texto_hipotesis, consulta, alto=260)

# ───────────────────────────────────────────── evaluación
with tab_eval:
    if metricas.empty:
        st.info("Aún no hay métricas: ejecuta 02_evaluacion.ipynb.")
    else:
        st.markdown("Todos los modelos se evaluaron sobre **los mismos 10 documentos y 10 clips** del corpus, con "
                    "referencias transcritas a mano y la misma normalización. Las configuraciones se fijaron antes "
                    "de ver los resultados (`modelos/config_evaluacion.json`).")
        t = metricas.copy()
        t["error"] = t["valor"].astype(float)
        st.dataframe(t[["modalidad", "modelo", "rol", "metrica", "error", "promedio_por_archivo", "segundos_por_item", "rtf", "descripcion"]]
                     .style.format({"error": "{:.1%}", "promedio_por_archivo": "{:.1%}", "segundos_por_item": "{:.1f}", "rtf": "{:.2f}"}, na_rep="—"),
                     hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            e = ev_doc[ev_doc["tipo"].isin(tipos_sel)]
            fig = px.box(e, x="tipo", y="cer", color="modelo", points="all", category_orders={"tipo": TIPOS},
                         labels={"cer": "CER por documento"}, title="CER por tipo de documento y modelo")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
        with c2:
            e = ev_aud[ev_aud["condicion"].isin(cond_sel)]
            fig = px.box(e, x="condicion", y="wer", color="modelo", points="all",
                         labels={"wer": "WER por clip", "condicion": "condición"}, title="WER por condición de grabación y modelo")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
        ma = metricas[metricas["modalidad"] == "audio"]
        if not ma.empty:
            fig = px.scatter(ma, x="rtf", y="valor", text="modelo", labels={"rtf": "factor de tiempo real (CPU)", "valor": "WER"},
                             title="Precisión vs. latencia de los modelos ASR")
            fig.update_traces(textposition="top center", marker_size=14)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch")
        if not auditoria.empty:
            st.subheader("Auditoría humana del LLM (20 extracciones al azar)")
            st.dataframe(auditoria[["id", "modalidad", "ok_modalidad_tema_resumen", "ok_campos_con_valor", "ok_nulls",
                                    "completamente_correcta", "error_observado"]], hide_index=True, width="stretch")

# ───────────────────────────────────────────── datos y descarga
with tab_datos:
    st.markdown("Tablas filtradas según la barra lateral. Todos los textos están **revisados y seudonimizados**.")
    vista = st.radio("Tabla", ["Documentos", "Transcripciones", "Extracción LLM", "Fuentes y licencias"], horizontal=True)
    if vista == "Documentos":
        t = d_vis[["id", "tipo", "titulo", "fuente", "texto", "modelo_ocr", "texto_referencia"]].rename(columns={"texto": "texto_ocr"})
    elif vista == "Transcripciones":
        t = a_vis[["id", "condicion", "duracion_seg", "idioma", "texto", "modelo_asr", "n_hablantes", "texto_referencia"]].rename(columns={"texto": "texto_asr"})
    elif vista == "Extracción LLM":
        t = llm[llm["id"].isin(set(d_vis["id"]) | set(a_vis["id"]))] if not llm.empty else llm
    else:
        t = fuentes
    st.dataframe(t, hide_index=True, width="stretch")
    st.download_button("⬇️ Descargar CSV", t.to_csv(index=False).encode("utf-8"),
                       file_name=f"{vista.lower().replace(' ', '_')}.csv", mime="text/csv")

st.divider()
st.caption("Fuentes: Wikimedia Commons (dominio público, CC0, CC BY, CC BY-SA) y OpenSLR 73 (Google, CC BY-SA 4.0). "
           "Atribución de cada archivo en la pestaña «Datos y fuentes». Tarea 2 · Ciencia de Datos con Python · UP 2026-II.")
