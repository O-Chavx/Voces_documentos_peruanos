# Voces y documentos peruanos: del archivo escaneado al texto buscable

**Pregunta:** ¿pueden convertirse documentos escaneados y entrevistas en audio en un archivo de texto estructurado y buscable?

**Dashboard público:** https://voces-documentos.streamlit.app/ · **Estudiante:** Oscar Andrés Chavez Inocente (@O-Chavx)

**Archivos principales:** [01_extraccion.ipynb](tarea_2/01_extraccion.ipynb) · [02_evaluacion.ipynb](tarea_2/02_evaluacion.ipynb) · [app.py](tarea_2/app.py) · [datos/](tarea_2/datos) · [evidencias_ia/](tarea_2/evidencias_ia) · todo el trabajo está en la carpeta [`tarea_2/`](tarea_2)

Tarea 2 del curso Ciencia de Datos con Python (UP 2026-II). Buena parte de la memoria documental y oral del Perú no se encuentra en bases de datos, sino en resoluciones impresas, placas y carteles fotografiados en la calle, libros parroquiales manuscritos y entrevistas grabadas; es así que el presente trabajo construye un pipeline de inteligencia artificial que extrae el texto de documentos peruanos mediante OCR, transcribe entrevistas en español peruano mediante ASR, seudonimiza a las personas privadas, extrae campos estructurados con un modelo de lenguaje y publica los resultados en un dashboard de Streamlit. De esta manera, se busca evaluar no solo si el proceso es posible, sino también qué tan confiable resulta según el tipo de documento y las condiciones de grabación.

## Resultados principales

**Resultados Test: WER: 0.17 | CER: 0.61 | Acuerdo LLM-Humano: 17/20**

| Modalidad | Modelo | Métrica (10 archivos) | Latencia en CPU |
|---|---|---|---|
| Documentos | PaddleOCR 3.7 · PP-OCRv6 (línea base) | CER 0,653 | 39,8 s por página |
| Documentos | PaddleOCR con orientación y *unwarping* | CER 0,622 | 32,4 s por página |
| Documentos | **EasyOCR 1.7 (elegido)** | **CER 0,606** | **6,5 s por página** |
| Audio | faster-whisper `small` (línea base) | WER 0,209 | factor de tiempo real 0,11 |
| Audio | **faster-whisper `large-v3-turbo` (elegido)** | **WER 0,168** | factor de tiempo real 0,19 |
| Audio | Qwen3-ASR 0.6B | WER 0,214 | factor de tiempo real 0,31 |
| Extracción LLM | Gemini Flash (3.8, 3.6 y 3.5-lite) | **17/20** fichas completamente correctas (documentos 8/10, audios 9/10) | — |

### Hallazgos

1. **El tipo de material pesa más que el modelo elegido.** En audio, la lectura en voz alta de OpenSLR 73 se transcribe casi sin errores (WER 0–4 %) y la mayoría de entrevistas reales se mantiene entre 4 % y 8 %; sin embargo, en una entrevista donde la persona habla bajo y lejos del micrófono (clip_010) los tres modelos omiten cerca del 85 % de lo dicho, porque el detector de voz trata su voz como silencio. En documentos ocurre algo similar, ya que las placas fotografiadas en la calle se leen con un CER de 8–11 % con PaddleOCR, mientras que una partida de bautismo manuscrita supera el 100 %, debido a que el OCR genera más caracteres de ruido que de texto real. Es por ello que la confianza mediana del OCR, que cae de 1,00 en impresos a 0,77 en manuscritos, resulta útil como alarma para enviar documentos a revisión humana.
2. **Un modelo más grande o un preprocesamiento adicional no garantizan una mejora.** Whisper `large-v3-turbo` reduce el WER en 4 puntos frente a la línea base y continúa siendo cinco veces más rápido que el tiempo real en una laptop sin GPU; no obstante, Qwen3-ASR, de una arquitectura distinta, no mejora a la línea base en español peruano. Asimismo, el *unwarping* de PaddleOCR ayudó en un cartel con muchos bloques de texto, pero deformó una placa que ya estaba plana (CER de 8 % a 71 %). En la extracción con Gemini, los tres errores encontrados en la auditoría son de naturaleza distinta (una omisión, una "corrección" no solicitada de un error del OCR y una alucinación), y ninguno es detectado por el chequeo automático de citas literales.

### Limitaciones

1. **La muestra de evaluación es pequeña y el CER depende de la regla de transcripción.** Diez archivos por modalidad permiten comparar modelos, pero no estimar el error con precisión: un solo documento (doc_026) o un solo clip (clip_010) mueve el resultado agregado varios puntos. Además, en los billetes de intis la referencia manual registró solo el texto principal, mientras que el OCR también lee la serie repetida y los cargos de los firmantes, por lo que esas inserciones elevan el CER de los impresos (59–91 %) aun cuando la lectura del texto principal es correcta.
2. **El procesamiento se realizó solo en CPU y la automatización tiene límites.** Por la ausencia de GPU no se evaluaron modelos de visión-lenguaje (PaddleOCR-VL), que leen mejor la escritura a mano, ni Tesseract, que no estaba instalado en el equipo. La diarización sobresegmenta las grabaciones con multitudes (11 "hablantes" en un clip de una marcha), y la seudonimización automática no reconoció todos los nombres, por lo que fue necesaria una revisión manual adicional.

## Corpus y fuentes

Los archivos se descargaron el **25 de octubre de 2026** con `src/descargar.py`, que consulta la API de Wikimedia Commons con un User-Agent descriptivo y pausas de 2 s entre solicitudes, y que acepta únicamente licencias libres (sin cláusulas NC ni ND) y archivos con categorías peruanas. La URL, la licencia, el autor y la forma de recuperar cada archivo se encuentran en [`datos/fuentes.csv`](tarea_2/datos/fuentes.csv).

| Corpus | Volumen | Composición | Licencias |
|---|---|---|---|
| **A. Documentos** | 25 imágenes | 8 impresos (resolución municipal, *Mercurio Peruano* de 1791, billetes de intis), 8 fotos de placas y carteles con sombra o inclinación, 9 manuscritos (Acta de la Independencia de 1821, tarjeta de Ricardo Palma, libros parroquiales de Lampa y Pisco, siglos XVII–XX) | 9 CC0, 5 dominio público, 6 CC BY-SA 4.0, 2 CC BY-SA 3.0, 1 CC BY-SA 2.0, 1 CC BY 4.0, 1 CC BY 3.0 |
| **B. Audio** | 30 clips · 38,0 min | 18 clips de 8 entrevistas reales de Commons (calle, marchas, fiestas patrias, organizaciones de mujeres indígenas; 23,1 min), 11 sesiones de lectura de [OpenSLR 73](https://www.openslr.org/73/) (13,3 min) y 1 entrevista simulada de 2 hablantes (1,6 min); 15 clips con 2 o más hablantes según la diarización | 30 CC BY-SA 4.0 |

- **Base en el curso:** los documentos verificados provienen de la clase de PaddleOCR (Sesión 10) y las entrevistas, OpenSLR 73 y la entrevista simulada, de `datos.py` (Sesión 13); `src/voz.py` es una copia del módulo de la Sesión 13.
- **Descartes:** 5 documentos con su motivo en [`datos/descartes.csv`](tarea_2/datos/descartes.csv) (2 ilegibles para el OCR, un casi duplicado, uno no peruano y una portada impresa mal clasificada como manuscrito). Ningún audio se descartó por falta de voz.
- **Seudonimización:** se reemplazaron 24 menciones de personas privadas (entrevistadas, entrevistadores y nombres en los títulos de Commons) por etiquetas como `[PERSONA_3]`; los nombres de figuras históricas y públicas (Ricardo Palma, firmantes del Acta de 1821) se conservan. El diccionario con los nombres reales no se versiona.
- **Muestra publicable** (`datos/ejemplos/`): miniaturas de los documentos con licencia libre y los clips de OpenSLR 73; las entrevistas de Commons no se redistribuyen y se consultan como texto seudonimizado.

<details>
<summary><b>Fuente, licencia y autor de cada archivo</b></summary>

| id | Tipo o condición | Descripción | Licencia | Autor | Fuente |
|---|---|---|---|---|---|
| doc_001 | impreso | Resolución de alcaldía de Mache (La Libertad), 2016 | CC BY 4.0 | Helder Zavaleta Verde | [Commons](https://commons.wikimedia.org/wiki/File:Acta_de_Creaci%C3%B3n_de_los_S%C3%ADmbolos_para_el_Distrito_de_Mache.jpg) |
| doc_002 | impreso | Mercurio Peruano, 1791 (imprenta colonial) | Public domain | El Mercurio Peruano | [Commons](https://commons.wikimedia.org/wiki/File:Pisco-punch-mercurio-peruano-1791.jpg) |
| doc_003 | impreso | Billete de 10 intis, 1987 | CC0 | ByRussia45 | [Commons](https://commons.wikimedia.org/wiki/File:10_Intis_1987(front).jpg) |
| doc_004 | impreso | Billete de 100 intis, 1987 | CC0 | ByRussia45 | [Commons](https://commons.wikimedia.org/wiki/File:100_Intis_1987(Front).jpg) |
| doc_005 | impreso | Billete de 1000 intis, 1988 | CC0 | ByRussia45 | [Commons](https://commons.wikimedia.org/wiki/File:1000_Intis_1988(Front).jpg) |
| doc_006 | impreso | Billete de 10 000 intis, 1988 | CC0 | ByRussia45 | [Commons](https://commons.wikimedia.org/wiki/File:10000_Intis_1988(Front).jpg) |
| doc_007 | impreso | Billete de 500 000 intis, 1988 | CC0 | ByRussia45 | [Commons](https://commons.wikimedia.org/wiki/File:500000_Intis_1988(Front).jpg) |
| doc_008 | impreso | Billete de 5 000 000 intis, 1990 | CC0 | ByRussia45 | [Commons](https://commons.wikimedia.org/wiki/File:5000000_Intis_1990(Front).jpg) |
| doc_010 | foto | Placa de la calle Capón, Barrio Chino de Lima (foto en la calle) | CC BY 3.0 | Miguel Angel Chong | [Commons](https://commons.wikimedia.org/wiki/File:Placa_Calle_Capon.jpg) |
| doc_012 | foto | Carteles chicha en Lima (foto en la calle) | CC BY-SA 2.0 | Lucía Calderon Portugal from Lima, Perú | [Commons](https://commons.wikimedia.org/wiki/File:Carteles_chicha.jpg) |
| doc_013 | manuscrito | Acta de la Independencia del Perú, 1821 | Public domain | Manuel Perez de Tudela | [Commons](https://commons.wikimedia.org/wiki/File:Acta_de_Independencia_Peru.jpg) |
| doc_014 | manuscrito | Tarjeta manuscrita de Ricardo Palma, 1912 | Public domain | Leguicall | [Commons](https://commons.wikimedia.org/wiki/File:Tarjeta_de_Ricardo_Palma_a_Luis_Antonio_Eguiguren_Escudero_24_octubre_1912.png) |
| doc_015 | manuscrito | Recibo de la Peruvian Telephone Company llenado a mano, 1895 | Public domain | Peruvian Telephone Company | [Commons](https://commons.wikimedia.org/wiki/File:Peruvian_Telephone_Company_receipt_Lima_1895.JPG) |
| doc_016 | manuscrito | Pizarra de menú escrita a mano con tiza, Lince, 2010 | CC BY-SA 3.0 | Dtarazona | [Commons](https://commons.wikimedia.org/wiki/File:Menu_Levaggi_09052010.jpg) |
| doc_017 | foto | Placa conmemorativa del Teatro Colón en Lima | CC BY-SA 4.0 | AlejandroFC | [Commons](https://commons.wikimedia.org/wiki/File:Placa_conmemorativa_del_Teatro_Col%C3%B3n_en_Lima.jpg) |
| doc_019 | foto | Pasaje Acisclo Villarán - Placa conmemorativa | CC BY-SA 4.0 | AlejandroFC | [Commons](https://commons.wikimedia.org/wiki/File:Pasaje_Acisclo_Villar%C3%A1n_-_Placa_conmemorativa.jpg) |
| doc_020 | foto | Placa de la Casa Federico Elguera, Calle La Virreyna, Lima 2 | CC BY-SA 4.0 | AlejandroFC | [Commons](https://commons.wikimedia.org/wiki/File:Placa_de_la_Casa_Federico_Elguera,_Calle_La_Virreyna,_Lima_2.jpg) |
| doc_021 | foto | Embassy of Hungary, Lima 3 | CC BY-SA 4.0 | Privado (imagen donada al publicador) | [Commons](https://commons.wikimedia.org/wiki/File:Embassy_of_Hungary,_Lima_3.jpg) |
| doc_022 | foto | Centro Español, Lima 01 | CC BY-SA 4.0 | AlejandroFC | [Commons](https://commons.wikimedia.org/wiki/File:Centro_Espa%C3%B1ol,_Lima_01.jpg) |
| doc_024 | foto | LimaPlacaCatedral1988 | CC BY-SA 3.0 | Miguel Angel Chong | [Commons](https://commons.wikimedia.org/wiki/File:LimaPlacaCatedral1988.JPG) |
| doc_025 | manuscrito | Cofradía de Nuestra Señora de la Soledad en la Villa de Pisco | CC0 | Vmpena100 | [Commons](https://commons.wikimedia.org/wiki/File:Cofrad%C3%ADa_de_Nuestra_Se%C3%B1ora_de_la_Soledad_en_la_Villa_de_Pisco.jpg) |
| doc_026 | manuscrito | Asiento de la partida de bautismo de Raúl Porras Barrenechea | Public domain | Partida N.° 2070. Asiento de la partida de ba | [Commons](https://commons.wikimedia.org/wiki/File:Asiento_de_la_partida_de_bautismo_de_Ra%C3%BAl_Porras_Barrenechea.jpg) |
| doc_027 | manuscrito | Joseolaya partida nacimiento chica | CC BY-SA 4.0 | Gabriela lavarello de velaochaga | [Commons](https://commons.wikimedia.org/wiki/File:Joseolaya_partida_nacimiento_chica.jpg) |
| doc_030 | manuscrito | Libro de Bautismos en la Gran Villa de Lampa 1832 | CC0 | Abelchristian111 | [Commons](https://commons.wikimedia.org/wiki/File:Libro_de_Bautismos_en_la_Gran_Villa_de_Lampa_1832.jpg) |
| doc_031 | manuscrito | Libro de Bautismos en la Gran Villa de Lampa 1871 | CC0 | Abelchristian111 | [Commons](https://commons.wikimedia.org/wiki/File:Libro_de_Bautismos_en_la_Gran_Villa_de_Lampa_1871.jpg) |
| clip_001–clip_002 | entrevista real (Commons) | Entrevista 1 en la Av. Rivagüero - El Agustino (136 s) | CC BY-SA 4.0 | JorgeFlores-pe | Commons (URL en `fuentes.csv`) |
| clip_003–clip_005 | entrevista real (Commons) | Entrevista 2 en la Av. Rivagüero - El Agustino (189 s) | CC BY-SA 4.0 | JorgeFlores-pe | Commons (URL en `fuentes.csv`) |
| clip_006–clip_008 | entrevista real (Commons) | Entrevista-[PERSONA_3]-organización de mujeres indígenas-fiestas patrias (251 s) | CC BY-SA 4.0 | QM Keen | Commons (URL en `fuentes.csv`) |
| clip_009 | entrevista real (Commons) | UNMSM Entrevista realizada en la marcha por [PERSONA_5] (77 s) | CC BY-SA 4.0 | Hellloff233 | Commons (URL en `fuentes.csv`) |
| clip_010 | entrevista real (Commons) | Entrevista a la señora [PERSONA_8], mujer shuyasina (98 s) | CC BY-SA 4.0 | Rashelley Perez | Commons (URL en `fuentes.csv`) |
| clip_011 | entrevista real (Commons) | Entrevista a la señora [PERSONA_6], mujer pomabambina (82 s) | CC BY-SA 4.0 | Rashelley Perez | Commons (URL en `fuentes.csv`) |
| clip_012 | entrevista real (Commons) | Audio de la entrevista a una chica que participaba de la marcha (54 s) | CC BY-SA 4.0 | Joaquin Canales | Commons (URL en `fuentes.csv`) |
| clip_013–clip_018 | entrevista real (Commons) | Entrevista a [PERSONA_7]-Organización de mujeres indígenas-fiestas patrias-protesta (498 s) | CC BY-SA 4.0 | QM Keen | Commons (URL en `fuentes.csv`) |
| clip_019 | lectura (OpenSLR 73) | OpenSLR 73 · hablante pef_01523 (mujer), 10 frases leídas (71 s) | CC BY-SA 4.0 | Google (Guevara-Rukoz et al., 2020) | [OpenSLR 73](https://www.openslr.org/73/) |
| clip_020 | lectura (OpenSLR 73) | OpenSLR 73 · hablante pem_00610 (hombre), 10 frases leídas (74 s) | CC BY-SA 4.0 | Google (Guevara-Rukoz et al., 2020) | [OpenSLR 73](https://www.openslr.org/73/) |
| clip_021 | lectura (OpenSLR 73) | OpenSLR 73 · hablante pef_00610 (mujer), 10 frases leídas (62 s) | CC BY-SA 4.0 | Google (Guevara-Rukoz et al., 2020) | [OpenSLR 73](https://www.openslr.org/73/) |
| clip_022 | lectura (OpenSLR 73) | OpenSLR 73 · hablante pem_05223 (hombre), 10 frases leídas (56 s) | CC BY-SA 4.0 | Google (Guevara-Rukoz et al., 2020) | [OpenSLR 73](https://www.openslr.org/73/) |
| clip_023 | lectura (OpenSLR 73) | OpenSLR 73 · hablante pef_01208 (mujer), 10 frases leídas (66 s) | CC BY-SA 4.0 | Google (Guevara-Rukoz et al., 2020) | [OpenSLR 73](https://www.openslr.org/73/) |
| clip_024 | lectura (OpenSLR 73) | OpenSLR 73 · hablante pem_07508 (hombre), 10 frases leídas (102 s) | CC BY-SA 4.0 | Google (Guevara-Rukoz et al., 2020) | [OpenSLR 73](https://www.openslr.org/73/) |
| clip_025 | lectura (OpenSLR 73) | OpenSLR 73 · hablante pef_07508 (mujer), 10 frases leídas (75 s) | CC BY-SA 4.0 | Google (Guevara-Rukoz et al., 2020) | [OpenSLR 73](https://www.openslr.org/73/) |
| clip_026 | lectura (OpenSLR 73) | OpenSLR 73 · hablante pem_09334 (hombre), 10 frases leídas (71 s) | CC BY-SA 4.0 | Google (Guevara-Rukoz et al., 2020) | [OpenSLR 73](https://www.openslr.org/73/) |
| clip_027 | lectura (OpenSLR 73) | OpenSLR 73 · hablante pef_02436 (mujer), 10 frases leídas (65 s) | CC BY-SA 4.0 | Google (Guevara-Rukoz et al., 2020) | [OpenSLR 73](https://www.openslr.org/73/) |
| clip_028 | lectura (OpenSLR 73) | OpenSLR 73 · hablante pem_03349 (hombre), 10 frases leídas (62 s) | CC BY-SA 4.0 | Google (Guevara-Rukoz et al., 2020) | [OpenSLR 73](https://www.openslr.org/73/) |
| clip_029 | lectura (OpenSLR 73) | OpenSLR 73 · hablante pef_06136 (mujer), 10 frases leídas (94 s) | CC BY-SA 4.0 | Google (Guevara-Rukoz et al., 2020) | [OpenSLR 73](https://www.openslr.org/73/) |
| clip_030 | multihablante (simulada) | Entrevista simulada: 2 hablantes de OpenSLR 73 alternando 14 turnos (98 s) | CC BY-SA 4.0 | Google (Guevara-Rukoz et al., 2020); montaje  | [OpenSLR 73](https://www.openslr.org/73/) |

</details>

## Método

| Etapa | Modelo base | Modelos alternativos |
|---|---|---|
| OCR | PaddleOCR 3.7 · PP-OCRv6 (motor PyTorch, CPU, sin preprocesamiento) | PaddleOCR con orientación de página, *unwarping* y orientación de línea · EasyOCR 1.7 (CRAFT + CRNN, `es`) |
| ASR | faster-whisper `small` (CPU int8, VAD Silero, `language="es"`) + diarización con WeSpeaker | faster-whisper `large-v3-turbo` · Qwen3-ASR 0.6B (mismos tramos del VAD) |
| Extracción | Gemini Flash, temperatura 0, salida JSON con esquema fijo y `null` para datos ausentes | si un modelo agota su cuota gratuita, el código pasa al siguiente (columna `modelo_llm`) |

La evaluación utiliza **10 documentos (4 impresos, 3 fotos, 3 manuscritos) y 10 clips (5 lecturas, 4 entrevistas reales y la entrevista simulada)** del propio corpus, transcritos a mano por el autor a partir del original. Las configuraciones, la normalización de textos y la regla de decisión se fijaron antes de calcular cualquier métrica (`modelos/config_evaluacion.json`), y todos los modelos se compararon sobre exactamente los mismos archivos.

## Estructura

```
tarea_2/
├── 01_extraccion.ipynb      # descarga, limpieza, OCR, ASR, diarización, seudonimización y EDA (Parte 1)
├── 02_evaluacion.ipynb      # muestra, CER/WER, modelos alternativos, LLM, auditoría y decisión (Parte 2)
├── app.py                   # dashboard de Streamlit (Parte 3)
├── requirements.txt         # dependencias del dashboard
├── src/
│   ├── descargar.py         # corpus desde Commons y OpenSLR 73
│   ├── t2.py                # OCR, métricas, seudonimización, Gemini y gráficos
│   └── voz.py               # utilidades de audio de la Sesión 13 (VAD, diarización, SRT/TXT)
├── entorno/                 # pyproject.toml + uv.lock del curso
├── modelos/config_evaluacion.json
├── figuras/                 # gráficos en PNG y HTML
├── evidencias_ia/           # capturas de la conversación con el asistente de IA
└── datos/
    ├── documentos.csv · transcripciones.csv · fuentes.csv · descartes.csv
    ├── extraccion_llm.csv · auditoria_llm.csv
    ├── metricas_modelos.csv · eval_documentos.csv · eval_audio.csv · resumen_resultados.json
    └── ejemplos/            # imágenes y audios publicables
```

## Cómo ejecutarlo

**Dashboard en local** (solo lee los CSV):

```bash
git clone https://github.com/O-Chavx/Voces_documentos_peruanos.git
cd Voces_documentos_peruanos/tarea_2
python -m venv .venv-app
.venv-app\Scripts\activate          # Windows; en macOS/Linux: source .venv-app/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Pipeline completo** (opcional; descarga cerca de 3 GB de modelos y corre en CPU), en PowerShell con [uv](https://docs.astral.sh/uv/):

```powershell
cd tarea_2\entorno
uv sync --group docai --group audio --group llm
uv pip install "plotly==7.0.0" kaleido easyocr "opencv-python==4.10.0.84" "opencv-python-headless==4.10.0.84" "opencv-contrib-python==4.10.0.84"
cd ..
copy .env.example .env        # y colocar GEMINI_API_KEY
.\entorno\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 01_extraccion.ipynb
.\entorno\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 02_evaluacion.ipynb
```

El notebook 02 se ejecuta en dos pasadas: la primera corre los modelos y el LLM y crea las plantillas de referencias manuales y de auditoría en `datos/raw/`; la segunda, una vez completadas por una persona, calcula las métricas.

## Uso de IA

El desarrollo se apoyó en Claude (Anthropic) como asistente de programación; los notebooks se ejecutaron en el equipo del autor, y las transcripciones de referencia y la auditoría del LLM se realizaron manualmente. Las capturas de la conversación se encuentran en `evidencias_ia/`.
