import os
import io
import zipfile
import tempfile
import traceback
import time

import streamlit as st
import numpy as np
import rasterio
import geopandas as gpd

# ============================================================
# PARCHE 1: compatibilidad NumPy 2.x <-> pysheds antiguo
# ============================================================
if not hasattr(np, "in1d"):     np.in1d     = np.isin
if not hasattr(np, "float_"):   np.float_   = np.float64
if not hasattr(np, "complex_"): np.complex_ = np.complex128
if not hasattr(np, "unicode_"): np.unicode_ = np.str_
if not hasattr(np, "object_"):  np.object_  = object
if not hasattr(np, "bool8"):    np.bool8    = np.bool_
if not hasattr(np, "int0"):     np.int0     = np.intp
if not hasattr(np, "NaN"):      np.NaN      = np.nan
if not hasattr(np, "Inf"):      np.Inf      = np.inf

# ============================================================
# PARCHE 2: memoryview -> ndarray (NumPy 2.x lo dejó de hacer)
# ============================================================
_asarray_original = np.asarray
def _asarray_patched(a, dtype=None, order=None, **kwargs):
    if isinstance(a, memoryview):
        a = np.frombuffer(a, dtype=a.format if a.format else None)
        if dtype is not None:
            a = a.astype(dtype, copy=False)
        return a
    return _asarray_original(a, dtype=dtype, order=order, **kwargs)
np.asarray = _asarray_patched

try:
    from pysheds.grid import Grid
    PYSHEDS_OK = True
except ImportError:
    PYSHEDS_OK = False


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
# CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    .main-title { color: #38bdf8; font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .subtitle { color: #94a3b8; font-size: 1rem; margin-top: 0; }
    .stButton>button { background-color: #0284c7; color: white; font-weight: 600; border: none; padding: 0.6rem 1.2rem; border-radius: 6px; }
    .stButton>button:hover { background-color: #0369a1; }
    .log-box { background-color: #0f172a; color: #34d399; font-family: 'Consolas', monospace; font-size: 0.85rem; padding: 1rem; border-radius: 6px; max-height: 400px; overflow-y: auto; white-space: pre-wrap; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def force_crs_on_vector(vector_path, crs):
    if not os.path.exists(vector_path):
        return None
    gdf = gpd.read_file(vector_path)
    gdf = gdf.set_crs(crs, allow_override=True)
    gdf.to_file(vector_path, driver="ESRI Shapefile")
    return gdf


def extract_stream_outlets(streams, acc, spacing):
    """Genera array con IDs únicos en cabeceras de segmento."""
    stream_mask = streams > 0
    stream_acc = np.where(stream_mask, acc, 0)

    rows, cols = np.where(stream_mask)
    if len(rows) == 0:
        raise ValueError("No hay celdas de río.")

    acc_vals = stream_acc[rows, cols]
    order = np.argsort(-acc_vals)
    rows = rows[order]
    cols = cols[order]

    sampled_rows = rows[::max(1, spacing)]
    sampled_cols = cols[::max(1, spacing)]

    outlets = np.zeros_like(streams, dtype=np.int32)
    for i, (r, c) in enumerate(zip(sampled_rows, sampled_cols), start=1):
        outlets[r, c] = i

    return outlets, len(sampled_rows)


def zip_outputs(folder):
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
        help="Cada N celdas de río se toma un punto de salida."
    )

    st.divider()
    run_btn = st.button("🚀 GENERAR MICROCUENCAS", use_container_width=True)


# ============================================================
# PROCESAMIENTO
# ============================================================
if not PYSHEDS_OK:
    st.error("⚠️ `pysheds` no está instalado. Revisá `requirements.txt`.")
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

            log("=== Inicio del procesamiento (pysheds) ===")
            update_log()
            progress.progress(5, text="Verificando CRS del DEM...")

            with rasterio.open(dem_path) as src:
                dem_crs = src.crs
                dem_shape = (src.height, src.width)

            if dem_crs is None:
                st.error("❌ El DEM no tiene CRS definido.")
                st.stop()

            log(f"CRS del DEM: {dem_crs}")
            log(f"Tamaño: {dem_shape[0]} x {dem_shape[1]} px")
            if not dem_crs.is_projected:
                log("⚠️ CRS no proyectado. Las áreas podrían ser incorrectas.")

            # --- 1. Cargar con pysheds
            log("[1/8] Cargando DEM en pysheds...")
            update_log(); progress.progress(10, text="Cargando DEM...")
            grid = Grid.from_raster(dem_path)
            dem = grid.read_raster(dem_path)

            # --- 2. Fill pits + depressions + flats
            log("[2/8] Rellenando depresiones...")
            update_log(); progress.progress(20, text="Rellenando depresiones...")
            pit_filled = grid.fill_pits(dem)
            flooded = grid.fill_depressions(pit_filled)
            inflated = grid.resolve_flats(flooded)

            # --- 3. Flow direction (D8)
            log("[3/8] Dirección de flujo (D8)...")
            update_log(); progress.progress(30, text="Dirección de flujo...")
            fdir = grid.flowdir(inflated)

            # --- 4. Flow accumulation
            log("[4/8] Acumulación de flujo...")
            update_log(); progress.progress(45, text="Acumulación de flujo...")
            acc = grid.accumulation(fdir)

            # --- 5. Streams por umbral
            log(f"[5/8] Red hídrica (umbral={threshold})...")
            update_log(); progress.progress(55, text="Extrayendo red hídrica...")
            acc_arr = np.asarray(acc)
            stream_mask = (acc_arr >= threshold).astype(np.uint8)
            n_cells = int(stream_mask.sum())
            log(f"Celdas de río: {n_cells}")
            if n_cells == 0:
                raise ValueError(
                    f"Umbral {threshold} muy alto. Reducilo (ej: 500)."
                )

            # Guardar ráster de streams
            streams_path = os.path.join(out_folder, "04_streams.tif")
            with rasterio.open(dem_path) as src:
                meta = src.meta.copy()
            meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
            with rasterio.open(streams_path, "w", **meta) as dst:
                dst.write(stream_mask, 1)

            # --- 6. Puntos de salida
            log("[6/8] Puntos de salida por segmento...")
            update_log(); progress.progress(65, text="Puntos de salida...")
            outlets_arr, n_outlets = extract_stream_outlets(
                stream_mask, acc_arr, spacing
            )
            log(f"Puntos de salida: {n_outlets}")

            outlets_path = os.path.join(out_folder, "06_outlets.tif")
            meta_out = meta.copy()
            meta_out.update({"dtype": "int32"})
            with rasterio.open(outlets_path, "w", **meta_out) as dst:
                dst.write(outlets_arr, 1)

            # --- 7. Watershed por cada punto de salida
            log("[7/8] Delimitando microcuencas...")
            update_log(); progress.progress(78, text="Delimitando cuencas...")

            # Obtener coordenadas (r, c) de cada punto de salida
            outlet_rows, outlet_cols = np.where(outlets_arr > 0)
            n_out = len(outlet_rows)
            log(f"Procesando {n_out} puntos...")

            all_catchments = []
            for i, (r, c) in enumerate(zip(outlet_rows, outlet_cols)):
                try:
                    catch = grid.catchment(x=int(c), y=int(r),
                                           fdir=fdir, xytype='index')
                    if catch is not None:
                        arr = np.asarray(catch).astype(np.uint8)
                        if arr.sum() > 0:
                            all_catchments.append((i + 1, arr))
                except Exception:
                    continue

                if (i + 1) % 20 == 0 or (i + 1) == n_out:
                    pct = 78 + int(10 * (i + 1) / n_out)
                    progress.progress(pct, text=f"Cuenca {i+1}/{n_out}...")

            log(f"Cuencas válidas: {len(all_catchments)}")
            if not all_catchments:
                raise ValueError("No se generó ninguna cuenca.")

            # --- 8. Vectorizar
            log("[8/8] Vectorizando...")
            update_log(); progress.progress(90, text="Vectorizando...")

            from rasterio.features import shapes as raster_shapes
            from shapely.geometry import shape

            # Combinar todas las cuencas en un solo array con IDs
            combined = np.zeros(dem_shape, dtype=np.int32)
            for bid, arr in all_catchments:
                combined[arr > 0] = bid

            # Guardar ráster de watersheds
            ws_path = os.path.join(out_folder, "07_watersheds.tif")
            meta_ws = meta.copy()
            meta_ws.update({"dtype": "int32"})
            with rasterio.open(ws_path, "w", **meta_ws) as dst:
                dst.write(combined, 1)

            # Red hídrica vectorial
            try:
                from rasterio.features import shapes as rs
                stream_shapes = list(rs(stream_mask, transform=grid.affine))
                features_s = []
                for shp, val in stream_shapes:
                    if val == 1:
                        try:
                            poly = shape(shp)
                            if poly.is_valid and not poly.is_empty:
                                features_s.append({"geometry": poly})
                        except Exception:
                            continue

                if features_s:
                    gdf_s = gpd.GeoDataFrame(features_s, crs=dem_crs)
                    gdf_s["ORDEN"] = 1  # pysheds no calcula Strahler directo
                    gdf_s.to_file(os.path.join(out_folder, "red_hidrica.geojson"),
                                  driver="GeoJSON")
                    try:
                        gdf_s.to_file(os.path.join(out_folder, "red_hidrica.shp"))
                    except Exception:
                        pass
                    log(f"Red hídrica: {len(gdf_s)} segmentos")
            except Exception as e:
                log(f"(Aviso) Red hídrica: {e}")

            # Microcuencas vectoriales
            try:
                ws_shapes = list(rs(combined.astype(np.int32), transform=grid.affine))
                features_w = []
                for shp, val in ws_shapes:
                    if val > 0:
                        try:
                            poly = shape(shp)
                            if poly.is_valid and not poly.is_empty:
                                features_w.append({
                                    "Microbasin": int(val),
                                    "geometry": poly
                                })
                        except Exception:
                            continue

                if features_w:
                    gdf_w = gpd.GeoDataFrame(features_w, crs=dem_crs)
                    gdf_w["Area_ha"] = gdf_w.geometry.area / 10000.0
                    gdf_w.to_file(os.path.join(out_folder, "microcuencas.geojson"),
                                  driver="GeoJSON")
                    try:
                        gdf_w.to_file(os.path.join(out_folder, "microcuencas.shp"))
                    except Exception:
                        pass
                    log(f"Microcuencas: {len(gdf_w)} cuencas")
            except Exception as e:
                log(f"(Aviso) Microcuencas: {e}")

            progress.progress(100, text="¡Listo!")
            log("=" * 50)
            log(f"PROCESO COMPLETADO — {len(all_catchments)} microcuencas")
            update_log()

            st.success(f"✅ {len(all_catchments)} microcuencas generadas.")

            # Vista previa
            try:
                gdf_preview = gpd.read_file(
                    os.path.join(out_folder, "microcuencas.geojson")
                )
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

            # ZIP
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
