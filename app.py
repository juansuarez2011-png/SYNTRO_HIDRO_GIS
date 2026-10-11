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
from scipy import ndimage
from rasterio.features import shapes as raster_shapes
from shapely.geometry import shape


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
    st.markdown('<p class="subtitle">Delimitación automática de microcuencas + red hídrica</p>', unsafe_allow_html=True)

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

    smooth_size = st.number_input(
        "Suavizado (vecinos)",
        min_value=3, max_value=31, value=9, step=2,
        help="Tamaño del kernel de suavizado. Mayor = formas más orgánicas."
    )

    st.divider()
    run_btn = st.button("🚀 GENERAR MICROCUENCAS", use_container_width=True)


# ============================================================
# PROCESAMIENTO
# ============================================================
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

            # Guardar DEM subido
            dem_path = os.path.join(tmp_dir, "dem_upload.tif")
            with open(dem_path, "wb") as f:
                f.write(dem_file.read())

            log("=== Inicio del procesamiento (modo liviano) ===")
            update_log()
            progress.progress(5, text="Verificando DEM...")

            with rasterio.open(dem_path) as src:
                dem_crs = src.crs
                transform = src.transform
                dem_arr = src.read(1).astype(np.float32)
                nodata = src.nodata
                shape_hw = (src.height, src.width)

            log(f"CRS: {dem_crs}")
            log(f"Tamaño: {shape_hw[0]} x {shape_hw[1]} px")

            if dem_crs is None:
                st.error("❌ El DEM no tiene CRS definido.")
                st.stop()

            # Manejar nodata
            if nodata is not None:
                dem_arr[dem_arr == nodata] = np.nan
            valid_mask = ~np.isnan(dem_arr)

            # --- 1. Suavizar DEM para reducir ruido
            log("[1/6] Suavizando DEM...")
            update_log(); progress.progress(15, text="Suavizando DEM...")

            dem_filled = np.where(valid_mask, dem_arr, np.nanmax(dem_arr))
            dem_smooth = ndimage.uniform_filter(dem_filled, size=int(smooth_size))

            # --- 2. Detectar cauces por acumulación local
            log("[2/6] Detectando cauces por mínimos locales...")
            update_log(); progress.progress(30, text="Detectando cauces...")

            # Filtro de mínimos: píxeles más bajos que sus vecinos = posibles cauces
            local_min = ndimage.minimum_filter(dem_smooth, size=int(smooth_size))
            topo_diff = dem_smooth - local_min

            # Umbral adaptativo
            thr_val = np.nanpercentile(topo_diff[valid_mask], 5)  # 5% más bajo
            stream_mask = ((topo_diff <= thr_val) & valid_mask).astype(np.uint8)

            n_stream_cells = int(stream_mask.sum())
            log(f"Celdas de cauce: {n_stream_cells}")
            if n_stream_cells == 0:
                raise ValueError("No se detectaron cauces. Probá otro suavizado.")

            # --- 3. Etiquetar microcuencas por proximidad a cauces
            log("[3/6] Etiquetando microcuencas...")
            update_log(); progress.progress(50, text="Etiquetando cuencas...")

            # Dilatar cada cauce para separar cuencas
            dilated = ndimage.binary_dilation(stream_mask, iterations=int(smooth_size))
            dilated = dilated & valid_mask

            # Etiquetar las zonas entre cauces
            labeled, n_zones = ndimage.label(valid_mask & ~dilated)
            log(f"Zonas detectadas: {n_zones}")

            # Asignar cada zona a la cuenca del cauce más cercano
            # Simplificación: cada zona se convierte en una microcuenca
            microcuencas = np.zeros_like(labeled, dtype=np.int32)

            # Filtrar zonas muy pequeñas
            min_zone_size = max(50, threshold // 10)
            zone_id = 1
            for i in range(1, n_zones + 1):
                mask = (labeled == i)
                if mask.sum() >= min_zone_size:
                    microcuencas[mask] = zone_id
                    zone_id += 1

            n_micro = zone_id - 1
            log(f"Microcuencas válidas: {n_micro}")

            if n_micro == 0:
                raise ValueError(
                    "No se generaron microcuencas. Bajá el umbral o el suavizado."
                )

            # --- 4. Vectorizar red hídrica
            log("[4/6] Vectorizando red hídrica...")
            update_log(); progress.progress(65, text="Vectorizando red hídrica...")

            try:
                stream_shapes = list(raster_shapes(stream_mask, transform=transform))
                features_s = []
                for shp, val in stream_shapes:
                    if val == 1:
                        try:
                            poly = shape(shp)
                            if poly.is_valid and not poly.is_empty:
                                features_s.append({"ORDEN": 1, "geometry": poly})
                        except Exception:
                            continue

                if features_s:
                    gdf_s = gpd.GeoDataFrame(features_s, crs=dem_crs)
                    gdf_s.to_file(os.path.join(out_folder, "red_hidrica.geojson"),
                                  driver="GeoJSON")
                    try:
                        gdf_s.to_file(os.path.join(out_folder, "red_hidrica.shp"))
                    except Exception:
                        pass
                    log(f"Red hídrica: {len(gdf_s)} segmentos")
            except Exception as e:
                log(f"(Aviso) Red hídrica: {e}")

            # --- 5. Vectorizar microcuencas
            log("[5/6] Vectorizando microcuencas...")
            update_log(); progress.progress(80, text="Vectorizando cuencas...")

            try:
                mc_shapes = list(raster_shapes(microcuencas.astype(np.int32),
                                               transform=transform))
                features_mc = []
                for shp, val in mc_shapes:
                    if val > 0:
                        try:
                            poly = shape(shp)
                            if poly.is_valid and not poly.is_empty:
                                features_mc.append({
                                    "Microbasin": int(val),
                                    "geometry": poly
                                })
                        except Exception:
                            continue

                if features_mc:
                    gdf_mc = gpd.GeoDataFrame(features_mc, crs=dem_crs)
                    gdf_mc["Area_ha"] = gdf_mc.geometry.area / 10000.0
                    gdf_mc.to_file(os.path.join(out_folder, "microcuencas.geojson"),
                                   driver="GeoJSON")
                    try:
                        gdf_mc.to_file(os.path.join(out_folder, "microcuencas.shp"))
                    except Exception:
                        pass
                    log(f"Microcuencas: {len(gdf_mc)} cuencas")
            except Exception as e:
                log(f"(Aviso) Microcuencas: {e}")

            # --- 6. Guardar rásters
            log("[6/6] Guardando rásters...")
            update_log(); progress.progress(90, text="Guardando...")

            with rasterio.open(dem_path) as src:
                meta = src.meta.copy()
            meta.update({"count": 1, "dtype": "uint8", "nodata": 0})

            with rasterio.open(os.path.join(out_folder, "streams.tif"),
                               "w", **meta) as dst:
                dst.write(stream_mask, 1)

            meta_i = meta.copy()
            meta_i.update({"dtype": "int32"})
            with rasterio.open(os.path.join(out_folder, "microcuencas.tif"),
                               "w", **meta_i) as dst:
                dst.write(microcuencas, 1)

            progress.progress(100, text="¡Listo!")
            log("=" * 50)
            log(f"PROCESO COMPLETADO — {n_micro} microcuencas")
            update_log()

            st.success(f"✅ {n_micro} microcuencas generadas.")

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

            # ZIP descarga
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
