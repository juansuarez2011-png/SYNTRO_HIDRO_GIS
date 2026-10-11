import os
import time
import tempfile
import traceback
import streamlit as st
import numpy as np
import rasterio
import geopandas as gpd

try:
    import whitebox
    WHITEBOX_OK = True
except ImportError:
    WHITEBOX_OK = False

st.set_page_config(
    page_title="Syntro GIS - Delimitador Hidrográfico",
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
        st.title("SINTRO HIDRO-GIS: Microcuencas + Orden de Ríos (UTM)")
    
    st.markdown("Herramienta automatizada de análisis hidrológico.")

    if not WHITEBOX_OK:
        st.error("WhiteboxTools no está instalado.")
        return

    with st.sidebar:
        if os.path.exists("icon.png"):
            st.image("icon.png", width=120)
        st.header("⚙️ Parámetros de Análisis")
        threshold = st.number_input("Umbral de Acumulación (Celdas)", min_value=10, max_value=50000, value=500, step=50)
        spacing = st.number_input("Puntos de Salida por Río (Espaciado)", min_value=10, max_value=2000, value=200, step=50)

    uploaded_file = st.file_uploader("📂 Sube tu Modelo Digital de Elevación (DEM en formato GeoTIFF)", type=["tif", "tiff"])

    if uploaded_file is not None:
        st.success(f"Archivo cargado correctamente: {uploaded_file.name}")
        
        if st.button("🚀 EJECUTAR PIPELINE HIDROLÓGICO", type="primary"):
            log_container = st.empty()
            progress_bar = st.progress(0)
            log_messages = []

            def log(msg):
                timestamp = time.strftime('%H:%M:%S')
                line = f"[{timestamp}] {msg}"
                log_messages.append(line)
                log_container.text_area("Registro de Ejecución (Log)", value="\n".join(log_messages), height=250)

            with tempfile.TemporaryDirectory() as tmp_dir:
                dem_input_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(dem_input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    start_time = time.time()
                    log("=== Iniciando Pipeline Hidrológico (CRS UTM) ===")
                    progress_bar.progress(5)

                    with rasterio.open(dem_input_path) as src:
                        dem_crs = src.crs
                    if dem_crs is None:
                        raise ValueError("El DEM no tiene CRS definido. Asignale un sistema proyectado UTM.")
                    log(f"CRS detectado en el DEM: {dem_crs}")

                    # Solución robusta de permisos para WhiteboxTools en Linux
                    wbt = whitebox.WhiteboxTools()
                    wbt.set_verbose_mode(False)
                    wbt.work_dir = tmp_dir
                    wbt.set_working_dir(tmp_dir)
                    
                    # Forzar la ruta del ejecutable al directorio temporal si es necesario
                    wbt.exe_path = os.path.join(tmp_dir, "whitebox_tools")

                    dem_fill         = os.path.join(tmp_dir, "01_dem_fill.tif")
                    dem_fdir         = os.path.join(tmp_dir, "02_flow_dir.tif")
                    dem_acc          = os.path.join(tmp_dir, "03_flow_accum.tif")
                    dem_stream       = os.path.join(tmp_dir, "04_streams.tif")
                    dem_stream_order = os.path.join(tmp_dir, "05_stream_order.tif")
                    dem_outlets      = os.path.join(tmp_dir, "06_outlets.tif")
                    dem_watershed    = os.path.join(tmp_dir, "07_watersheds.tif")
                    streams_vec      = os.path.join(tmp_dir, "red_hidrica.shp")
                    watersheds_vec   = os.path.join(tmp_dir, "microcuencas.shp")
                    streams_geojson  = os.path.join(tmp_dir, "red_hidrica.geojson")
                    watersheds_geojson = os.path.join(tmp_dir, "microcuencas.geojson")

                    log("[1/8] Rellenando depresiones topográficas...")
                    progress_bar.progress(15)
                    wbt.fill_depressions(dem_input_path, dem_fill)
                    force_crs_on_raster(dem_fill, dem_crs)

                    log("[2/8] Calculando dirección de flujo (D8)...")
                    progress_bar.progress(30)
                    wbt.d8_pointer(dem_fill, dem_fdir)
                    force_crs_on_raster(dem_fdir, dem_crs)

                    log("[3/8] Calculando acumulación de flujo...")
                    progress_bar.progress(45)
                    wbt.d8_flow_accumulation(dem_fill, dem_acc, out_type='cells')
                    force_crs_on_raster(dem_acc, dem_crs)

                    log(f"[4/8] Extrayendo red hídrica con umbral = {threshold}...")
                    progress_bar.progress(60)
                    wbt.extract_streams(dem_acc, dem_stream, threshold=threshold)
                    force_crs_on_raster(dem_stream, dem_crs)

                    with rasterio.open(dem_stream) as src:
                        streams_arr = src.read(1)
                        n_cells = int((streams_arr > 0).sum())
                    log(f"  Celdas de río detectadas: {n_cells}")
                    if n_cells == 0:
                        raise ValueError(f"El umbral {threshold} es muy alto y no generó red hídrica.")

                    log("[5/8] Calculando orden de ríos (Strahler)...")
                    progress_bar.progress(70)
                    try:
                        wbt.strahler_stream_order(dem_fdir, dem_stream, dem_stream_order)
                        force_crs_on_raster(dem_stream_order, dem_crs)
                        log("  Orden Strahler calculado con éxito.")
                    except Exception as e:
                        log(f"  (Aviso) Orden Strahler omitido: {e}")
                        dem_stream_order = None

                    log("[6/8] Extrayendo puntos de salida (outlets)...")
                    progress_bar.progress(80)
                    n_outlets = extract_stream_outlets(dem_stream, dem_fdir, dem_acc, dem_outlets, spacing)
                    force_crs_on_raster(dem_outlets, dem_crs)
                    log(f"  Puntos de salida generados: {n_outlets}")

                    log("[7/8] Delimitando microcuencas...")
                    progress_bar.progress(90)
                    wbt.watershed(dem_fdir, dem_outlets, dem_watershed, esri_pntr=False)
                    force_crs_on_raster(dem_watershed, dem_crs)

                    log("[8/8] Vectorizando resultados a formato GeoJSON...")
                    progress_bar.progress(95)

                    try:
                        wbt.raster_streams_to_vector(dem_stream, dem_fdir, streams_vec)
                        gdf_s = force_crs_on_vector(streams_vec, dem_crs)
                        if gdf_s is not None:
                            gdf_s.to_file(streams_geojson, driver="GeoJSON")
                    except Exception as e:
                        log(f"  (Aviso) Red hídrica vectorial: {e}")

                    try:
                        wbt.raster_to_vector_polygons(dem_watershed, watersheds_vec)
                        gdf_w = force_crs_on_vector(watersheds_vec, dem_crs)
                        if gdf_w is not None:
                            gdf_w["Area_ha"] = gdf_w.geometry.area / 10000.0
                            gdf_w.to_file(watersheds_geojson, driver="GeoJSON")
                    except Exception as e:
                        log(f"  (Aviso) Microcuencas vectoriales: {e}")

                    progress_bar.progress(100)
                    elapsed_time = int(time.time() - start_time)
                    log(f"🎉 ¡PROCESO COMPLETADO EN {elapsed_time}s — {n_outlets} microcuencas!")

                    st.success(f"Se procesaron {n_outlets} microcuencas correctamente.")

                    col1, col2 = st.columns(2)
                    if os.path.exists(watersheds_geojson):
                        with open(watersheds_geojson, "rb") as f:
                            col1.download_button("📥 Descargar Microcuencas (GeoJSON)", f, "microcuencas.geojson", "application/geo+json")
                    
                    if os.path.exists(streams_geojson):
                        with open(streams_geojson, "rb") as f:
                            col2.download_button("📥 Descargar Red Hídrica (GeoJSON)", f, "red_hidrica.geojson", "application/geo+json")

                except Exception as e:
                    st.error(f"Error durante el procesamiento: {str(e)}")
                    st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
