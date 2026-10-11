import os
import io
import zipfile
import tempfile
import traceback
import time

# ============================================================
# FIX PARA STREAMLIT CLOUD:
# whitebox intenta descargar su binario en site-packages (sin permisos).
# Le decimos que lo haga en /tmp/wbt (con permisos).
# ============================================================
os.environ["WBT_PATH"] = "/tmp/wbt"
os.makedirs("/tmp/wbt", exist_ok=True)

import streamlit as st
import numpy as np
import rasterio
import geopandas as gpd

try:
    import whitebox
    WHITEBOX_OK = True
except ImportError:
    WHITEBOX_OK = False


LOGO_PATH = "icon.png"


# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="SYNTRO Hidro-GIS",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS PERSONALIZADO
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    .main-title {
        color: #38bdf8;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0;
    }
    .stButton>button {
        background-color: #0284c7;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
    }
    .stButton>button:hover { background-color: #0369a1; }
    .log-box {
        background-color: #0f172a;
        color: #34d399;
        font-family: 'Consolas', monospace;
        font-size: 0.85rem;
        padding: 1rem;
        border-radius: 6px;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def force_crs_on_raster(raster_path, crs):
    """Reescribe un ráster forzando su CRS si no lo tiene."""
    if not os.path.exists(raster_path):
        return
    with rasterio.open(raster_path) as src:
        if src.crs is not None:
            return
        data = src.read()
        meta = src.meta.copy()
        meta.update({"crs": crs})
    with rasterio.open(raster_path, "w", **meta) as dst:
        dst.write(data)


def force_crs_on_vector(vector_path, crs):
    """Reescribe un shapefile forzando su CRS."""
    if not os.path.exists(vector_path):
        return None
    gdf = gpd.read_file(vector_path)
    gdf = gdf.set_crs(crs, allow_override=True)
    gdf.to_file(vector_path, driver="ESRI Shapefile")
    return gdf


def extract_stream_outlets(streams_path, acc_path, out_path, spacing):
    """Genera raster con IDs únicos en cabeceras de segmento de río."""
    with rasterio.open(streams_path) as src:
        streams = src.read(1)
        meta = src.meta.copy()

    with rasterio.open(acc_path) as src:
        acc = src.read(1).astype(np.float64)

    stream_mask = streams > 0
    stream_acc = np.where(stream_mask, acc, 0)

    rows, cols = np.where(stream_mask)
    if len(rows) == 0:
        raise ValueError("No hay celdas de río para extraer puntos de salida.")

    acc_vals = stream_acc[rows, cols]
    order = np.argsort(-acc_vals)
    rows = rows[order]
    cols = cols[order]

    sampled_rows = rows[::max(1, spacing)]
    sampled_cols = cols[::max(1, spacing)]

    outlets = np.zeros_like(streams, dtype=np.int32)
    for i, (r, c) in enumerate(zip(sampled_rows, sampled_cols), start=1):
        outlets[r, c] = i

    meta.update({"count": 1, "dtype": "int32", "nodata": 0})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(outlets, 1)

    return len(sampled_rows)


def zip_outputs(folder):
    """Empaqueta los archivos de salida en un ZIP para descarga."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder)
                zf.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer


# ============================================================
# HEADER
# ============================================================
col_logo, col_title = st.columns([1, 5])
with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=110)
with col_title:
    st.markdown('<h1 class="main-title">SYNTRO Hidro-GIS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Delimitación automática de microcuencas + red hídrica con orden de Strahler</p>', unsafe_allow_html=True)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)

    st.header("⚙️ Parámetros")

    dem_file = st.file_uploader(
        "📂 DEM de Elevación (GeoTIFF)",
        type=["tif", "tiff"],
        help="El DEM debe tener CRS proyectado (idealmente UTM)."
    )

    threshold = st.number_input(
        "Umbral de Acumulación (Celdas)",
        min_value=10, max_value=100000, value=500, step=50,
        help="Menor umbral = más microcuencas. Mayor umbral = menos y más grandes."
    )

    spacing = st.number_input(
        "Puntos de Salida por Río",
        min_value=10, max_value=2000, value=200, step=10,
        help="Cada N celdas de río se toma un punto de salida (emula Stream Link)."
    )

    st.divider()
    run_btn = st.button("🚀 GENERAR MICROCUENCAS", use_container_width=True)


# ============================================================
# PROCESAMIENTO
# ============================================================
if not WHITEBOX_OK:
    st.error("⚠️ WhiteboxTools no está instalado. Revisá `requirements.txt`.")
    st.stop()

if run_btn:
    if dem_file is None:
        st.warning("⚠️ Cargá un DEM antes de continuar.")
        st.stop()

    log_messages = []
    def log(msg):
        log_messages.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    progress = st.progress(0, text="Iniciando...")
    log_placeholder = st.empty()

    def update_log():
        log_placeholder.markdown(
            '<div class="log-box">' + "\n".join(log_messages) + '</div>',
            unsafe_allow_html=True
        )

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_folder = os.path.join(tmp_dir, "output")
            os.makedirs(out_folder, exist_ok=True)

            dem_path = os.path.join(tmp_dir, "dem_upload.tif")
            with open(dem_path, "wb") as f:
                f.write(dem_file.read())

            log("=== Inicio del procesamiento ===")
            update_log()
            progress.progress(5, text="Verificando CRS del DEM...")

            with rasterio.open(dem_path) as src:
                dem_crs = src.crs
                dem_shape = (src.height, src.width)

            if dem_crs is None:
                st.error("❌ El DEM no tiene CRS definido. Asignale uno (idealmente UTM).")
                st.stop()

            log(f"CRS del DEM: {dem_crs}")
            log(f"Tamaño: {dem_shape[0]} x {dem_shape[1]} px")
            if not dem_crs.is_projected:
                log("⚠️ CRS no proyectado. Las áreas podrían ser incorrectas.")

            # --- Inicializar WhiteboxTools (descarga binario en /tmp/wbt)
            log("Inicializando WhiteboxTools...")
            update_log()
            try:
                wbt = whitebox.WhiteboxTools()
                wbt.set_verbose_mode(False)
                wbt.set_working_dir(out_folder)
                log("WhiteboxTools listo.")
            except Exception as e:
                log(f"ERROR al inicializar WhiteboxTools: {e}")
                update_log()
                st.error(f"❌ No se pudo inicializar WhiteboxTools: {e}")
                st.stop()

            dem_fill         = os.path.join(out_folder, "01_dem_fill.tif")
            dem_fdir         = os.path.join(out_folder, "02_flow_dir.tif")
            dem_acc          = os.path.join(out_folder, "03_flow_accum.tif")
            dem_stream       = os.path.join(out_folder, "04_streams.tif")
            dem_stream_order = os.path.join(out_folder, "05_stream_order.tif")
            dem_outlets      = os.path.join(out_folder, "06_outlets.tif")
            dem_watershed    = os.path.join(out_folder, "07_watersheds.tif")
            streams_vec      = os.path.join(out_folder, "red_hidrica.shp")
            watersheds_vec   = os.path.join(out_folder, "microcuencas.shp")

            # --- 1. Fill
            log("[1/8] Rellenando depresiones...")
            update_log(); progress.progress(10, text="Rellenando depresiones...")
            wbt.fill_depressions(dem_path, dem_fill)
            force_crs_on_raster(dem_fill, dem_crs)

            # --- 2. Flow Direction
            log("[2/8] Dirección de flujo...")
            update_log(); progress.progress(20, text="Dirección de flujo...")
            wbt.d8_pointer(dem_fill, dem_fdir)
            force_crs_on_raster(dem_fdir, dem_crs)

            # --- 3. Flow Accumulation
            log("[3/8] Acumulación de flujo...")
            update_log(); progress.progress(30, text="Acumulación de flujo...")
            wbt.d8_flow_accumulation(dem_fill, dem_acc, out_type='cells')
            force_crs_on_raster(dem_acc, dem_crs)

            # --- 4. Streams por umbral
            log(f"[4/8] Red hídrica (umbral={threshold})...")
            update_log(); progress.progress(45, text="Extrayendo red hídrica...")
            wbt.extract_streams(dem_acc, dem_stream, threshold=threshold)
            force_crs_on_raster(dem_stream, dem_crs)

            with rasterio.open(dem_stream) as src:
                streams_arr = src.read(1)
                n_cells = int((streams_arr > 0).sum())
            log(f"Celdas de río: {n_cells}")
            if n_cells == 0:
                raise ValueError(f"El umbral {threshold} no generó red. Reducilo.")

            # --- 5. Orden de Strahler
            log("[5/8] Orden de Strahler...")
            update_log(); progress.progress(55, text="Orden de Strahler...")
            try:
                wbt.strahler_stream_order(dem_fdir, dem_stream, dem_stream_order)
                force_crs_on_raster(dem_stream_order, dem_crs)
                log("Orden calculado.")
            except Exception as e:
                log(f"(Aviso) Orden: {e}")
                dem_stream_order = None

            # --- 6. Puntos de salida
            log("[6/8] Puntos de salida...")
            update_log(); progress.progress(65, text="Puntos de salida...")
            n_outlets = extract_stream_outlets(dem_stream, dem_acc, dem_outlets, spacing)
            force_crs_on_raster(dem_outlets, dem_crs)
            log(f"Puntos de salida: {n_outlets}")

            # --- 7. Watershed
            log("[7/8] Delimitando cuencas...")
            update_log(); progress.progress(78, text="Delimitando cuencas...")
            wbt.watershed(dem_fdir, dem_outlets, dem_watershed, esri_pntr=False)
            force_crs_on_raster(dem_watershed, dem_crs)

            # --- 8. Vectorizar
            log("[8/8] Vectorizando...")
            update_log(); progress.progress(90, text="Vectorizando...")

            # Red hídrica con orden
            try:
                wbt.raster_streams_to_vector(dem_stream, dem_fdir, streams_vec)
                gdf_s = force_crs_on_vector(streams_vec, dem_crs)

                if gdf_s is not None and dem_stream_order and os.path.exists(dem_stream_order):
                    with rasterio.open(dem_stream_order) as src:
                        order_arr = src.read(1)
                        order_transform = src.transform

                    def get_order(geom):
                        try:
                            p = geom.interpolate(0.5, normalized=True)
                            col, row = ~order_transform * (p.x, p.y)
                            row, col = int(row), int(col)
                            if 0 <= row < order_arr.shape[0] and 0 <= col < order_arr.shape[1]:
                                v = int(order_arr[row, col])
                                return v if v > 0 else 1
                        except Exception:
                            pass
                        return 1

                    gdf_s["ORDEN"] = gdf_s.geometry.apply(get_order)

                if gdf_s is not None:
                    gdf_s.to_file(streams_vec, driver="ESRI Shapefile")
                    gdf_s.to_file(os.path.join(out_folder, "red_hidrica.geojson"),
                                  driver="GeoJSON")
                    log(f"Red hídrica: {len(gdf_s)} segmentos")
            except Exception as e:
                log(f"(Aviso) Red hídrica: {e}")

            # Microcuencas
            try:
                wbt.raster_to_vector_polygons(dem_watershed, watersheds_vec)
                gdf_w = force_crs_on_vector(watersheds_vec, dem_crs)
                if gdf_w is not None:
                    gdf_w["Area_ha"] = gdf_w.geometry.area / 10000.0
                    gdf_w.to_file(watersheds_vec, driver="ESRI Shapefile")
                    gdf_w.to_file(os.path.join(out_folder, "microcuencas.geojson"),
                                  driver="GeoJSON")
                    log(f"Microcuencas: {len(gdf_w)} cuencas")
            except Exception as e:
                log(f"(Aviso) Microcuencas: {e}")

            progress.progress(100, text="¡Listo!")
            log("=" * 50)
            log(f"PROCESO COMPLETADO — {n_outlets} microcuencas")
            update_log()

            st.success(f"✅ {n_outlets} microcuencas generadas. Descargá los resultados abajo.")

            # Vista previa
            try:
                gdf_preview = gpd.read_file(os.path.join(out_folder, "microcuencas.geojson"))
                st.subheader("📊 Vista previa")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Microcuencas", len(gdf_preview))
                with col2:
                    st.metric("Área total (ha)", f"{gdf_preview['Area_ha'].sum():.2f}")
                with col3:
                    st.metric("Área media (ha)", f"{gdf_preview['Area_ha'].mean():.2f}")
            except Exception:
                pass

            # Descarga ZIP
            zip_buffer = zip_outputs(out_folder)
            st.download_button(
                label="⬇️ Descargar resultados (ZIP)",
                data=zip_buffer,
                file_name="syntro_microcuencas.zip",
                mime="application/zip",
                use_container_width=True
            )

    except Exception as e:
        tb = traceback.format_exc()
        log(f"ERROR: {e}")
        for line in tb.splitlines():
            log(f"  {line}")
        update_log()
        st.error(f"❌ Error: {e}")
