import os
import time
import tempfile
import traceback
import urllib.request
import tarfile
import zipfile
import subprocess
import streamlit as st
import numpy as np
import rasterio
import geopandas as gpd

# -------------------------------------------------------------------------
# DESCARGA Y CONFIGURACIÓN AUTOMÁTICA DE WHITEBOXTOOLS PARA LINUX (NUBE)
# -------------------------------------------------------------------------
tmp_dir = tempfile.gettempdir()
wbt_folder = os.path.join(tmp_dir, "WBT")
wbt_exe = os.path.join(wbt_folder, "whitebox_tools")

if not os.path.exists(wbt_exe):
    try:
        os.makedirs(wbt_folder, exist_ok=True)
        zip_path = os.path.join(tmp_dir, "wbt.zip")
        url = "https://www.whiteboxgeo.com/WBTLinux/WhiteboxTools_linux_amd64.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
        # Dar permisos de ejecución al binario en Linux
        if os.path.exists(wbt_exe):
            os.chmod(wbt_exe, 0o755)
    except Exception:
        pass

try:
    import whitebox
    WHITEBOX_OK = True
except ImportError:
    WHITEBOX_OK = False

# Configuración de la interfaz web corporativa de Syntro GIS
st.set_page_config(
    page_title="Syntro GIS - Delimitador Hidrográfico por Umbral (UTM)",
    page_icon="icon.png" if os.path.exists("icon.png") else "🌊",
    layout="wide"
)

def force_crs_on_raster(raster_path, crs):
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
    if not os.path.exists(vector_path):
        return None
    gdf = gpd.read_file(vector_path)
    gdf = gdf.set_crs(crs, allow_override=True)
    try:
        gdf.to_file(vector_path, driver="ESRI Shapefile")
    except Exception:
        base = vector_path.rsplit(".", 1)[0]
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            p = base + ext
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
            gdf.to_file(vector_path, driver="ESRI Shapefile")
    return gdf

def extract_stream_outlets(streams_path, fdir_path, acc_path, out_path, spacing):
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

def main():
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        if os.path.exists("icon.png"):
            st.image("icon.png", width=80)
    with col_title:
        st.title("SINTRO HIDRO-GIS: MICROCUENCAS + ORDEN DE RÍOS (UTM)")
    
    st.markdown("Herramienta automatizada de análisis hidrológico y delimitación por umbral.")

    if not WHITEBOX_OK:
        st.error("WhiteboxTools no está instalado en el entorno.")
        return

    with st.sidebar:
        if os.path.exists("icon.png"):
            st.image("icon.png", width=120)
        st.header("⚙️ Parámetros de Delimitación")
        threshold = st.number_input("Umbral de Acumulación (Celdas)", min_value=10, max_value=50000, value=500, step=50)
        spacing = st.number_input("Puntos de Salida por Río (Espaciado)", min_value=10, max_value=2000, value=200, step=50)
        st.info("💡 *(Umbral BAJO = más microcuencas | Umbral ALTO = menos microcuencas grandes)*")

    uploaded_file = st.file_uploader("📂 DEM de Elevación (GeoTIFF)", type=["tif", "tiff"])

    if uploaded_file is not None:
        st.success(f"Archivo cargado correctamente: {uploaded_file.name}")
        
        if st.button("🚀 GENERAR MICROCUENCAS + ORDEN DE RÍOS", type="primary"):
            log_container = st.empty()
            progress_bar = st.progress(0)
            log_messages = []

            def log(msg):
                timestamp = time.strftime('%H:%M:%S')
                line = f"[{timestamp}] {msg}"
                log_messages.append(line)
                log_container.text_area("Registro de Ejecución (Log)", value="\n".join(log_messages), height=250)

            with tempfile.TemporaryDirectory() as run_dir:
                dem_input_path = os.path.join(run_dir, uploaded_file.name)
                with open(dem_input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    start_time = time.time()
                    log("=== Microcuencas + Orden de Ríos (CRS UTM) ===")
                    progress_bar.progress(3)

                    with rasterio.open(dem_input_path) as src:
                        dem_crs = src.crs
                    if dem_crs is None:
                        raise ValueError("El DEM no tiene CRS definido. Asignale uno proyectado UTM.")
                    log(f"CRS del DEM: {dem_crs}")

                    wbt = whitebox.WhiteboxTools()
                    wbt.set_verbose_mode(False)
                    wbt.set_working_dir(run_dir)
                    if os.path.exists(wbt_exe):
                        wbt.set_executable_path(wbt_exe)

                    dem_fill         = os.path.join(run_dir, "01_dem_fill.tif")
                    dem_fdir         = os.path.join(run_dir, "02_flow_dir.tif")
                    dem_acc          = os.path.join(run_dir, "03_flow_accum.tif")
                    dem_stream       = os.path.join(run_dir, "04_streams.tif")
                    dem_stream_order = os.path.join(run_dir, "05_stream_order.tif")
                    dem_outlets      = os.path.join(run_dir, "06_outlets.tif")
                    dem_watershed    = os.path.join(run_dir, "07_watersheds.tif")
                    streams_vec      = os.path.join(run_dir, "red_hidrica.shp")
                    watersheds_vec   = os.path.join(run_dir, "microcuencas.shp")
                    streams_geojson  = os.path.join(run_dir, "red_hidrica.geojson")
                    watersheds_geojson = os.path.join(run_dir, "microcuencas.geojson")

                    log("[1/8] Rellenando depresiones...")
                    progress_bar.progress(10)
                    wbt.fill_depressions(dem_input_path, dem_fill)
                    force_crs_on_raster(dem_fill, dem_crs)

                    log("[2/8] Calculando dirección de flujo (D8)...")
                    progress_bar.progress(20)
                    wbt.d8_pointer(dem_fill, dem_fdir)
                    force_crs_on_raster(dem_fdir, dem_crs)

                    log("[3/8] Calculando acumulación de flujo...")
                    progress_bar.progress(35)
                    wbt.d8_flow_accumulation(dem_fill, dem_acc, out_type='cells')
                    force_crs_on_raster(dem_acc, dem_crs)

                    log(f"[4/8] Extrayendo red hídrica con umbral={threshold}...")
                    progress_bar.progress(45)
                    wbt.extract_streams(dem_acc, dem_stream, threshold=threshold)
                    force_crs_on_raster(dem_stream, dem_crs)

                    with rasterio.open(dem_stream) as src:
                        n_cells = int((src.read(1) > 0).sum())
                    log(f"  Celdas de río detectadas: {n_cells}")
                    if n_cells == 0:
                        raise ValueError(f"El umbral {threshold} no generó red. Reducilo.")

                    log("[5/8] Calculando orden de ríos (Strahler)...")
                    progress_bar.progress(55)
                    try:
                        wbt.strahler_stream_order(dem_fdir, dem_stream, dem_stream_order)
                        force_crs_on_raster(dem_stream_order, dem_crs)
                        log("  Orden de ríos calculado correctamente.")
                    except Exception as e:
                        log(f"  (Aviso) Orden Strahler omitido: {e}")
                        dem_stream_order = None

                    log("[6/8] Extrayendo puntos de salida por segmento...")
                    progress_bar.progress(65)
                    n_outlets = extract_stream_outlets(dem_stream, dem_fdir, dem_acc, dem_outlets, spacing)
                    force_crs_on_raster(dem_outlets, dem_crs)
                    log(f"  Puntos de salida generados: {n_outlets}")

                    log("[7/8] Delimitando cuenca por cada punto de salida...")
                    progress_bar.progress(80)
                    wbt.watershed(dem_fdir, dem_outlets, dem_watershed, esri_pntr=False)
                    force_crs_on_raster(dem_watershed, dem_crs)

                    log("[8/8] Vectorizando resultados y forzando CRS...")
                    progress_bar.progress(92)

                    try:
                        wbt.raster_streams_to_vector(dem_stream, dem_fdir, streams_vec)
                        gdf_s = force_crs_on_vector(streams_vec, dem_crs)
                    except Exception as e:
                        log(f"  (Aviso) Vectorización de ríos: {e}")
                        gdf_s = None

                    if gdf_s is not None and dem_stream_order is not None and os.path.exists(dem_stream_order):
                        try:
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
                            log("  Orden de ríos añadido a la red hídrica.")
                        except Exception as e:
                            log(f"  (Aviso) No se pudo adjuntar orden: {e}")
                            gdf_s["ORDEN"] = 1
                    elif gdf_s is not None:
                        gdf_s["ORDEN"] = 1

                    if gdf_s is not None:
                        try:
                            gdf_s.to_file(streams_geojson, driver="GeoJSON")
                        except Exception as e:
                            log(f"  (Aviso) Guardado GeoJSON de red: {e}")

                    try:
                        wbt.raster_to_vector_polygons(dem_watershed, watersheds_vec)
                        gdf_w = force_crs_on_vector(watersheds_vec, dem_crs)
                        if gdf_w is not None:
                            gdf_w["Area_ha"] = gdf_w.geometry.area / 10000.0
                            gdf_w.to_file(watersheds_geojson, driver="GeoJSON")
                            log(f"  Microcuencas: {watersheds_geojson} ({len(gdf_w)} cuencas)")
                    except Exception as e:
                        log(f"  (Aviso) Microcuencas: {e}")

                    progress_bar.progress(100)
                    elapsed_time = int(time.time() - start_time)
                    log(f"PROCESO COMPLETADO — {n_outlets} microcuencas")

                    st.success(f"Se generaron {n_outlets} microcuencas y la red hídrica con orden correctamente.")

                    col1, col2 = st.columns(2)
                    if os.path.exists(watersheds_geojson):
                        with open(watersheds_geojson, "rb") as f_w:
                            col1.download_button(
                                label="📥 Descargar Microcuencas (GeoJSON)",
                                data=f_w,
                                file_name="microcuencas.geojson",
                                mime="application/geo+json"
                            )
                    
                    if os.path.exists(streams_geojson):
                        with open(streams_geojson, "rb") as f_s:
                            col2.download_button(
                                label="📥 Descargar Red Hídrica (GeoJSON)",
                                data=f_s,
                                file_name="red_hidrica.geojson",
                                mime="application/geo+json"
                            )

                except Exception as e:
                    st.error(f"Error durante el procesamiento: {str(e)}")
                    st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
