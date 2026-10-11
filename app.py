import os
import time
import tempfile
import traceback
import streamlit as st
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
from scipy.ndimage import minimum_filter, label

st.set_page_config(
    page_title="Syntro GIS - Mosaico de Microcuencas (UTM)",
    page_icon="icon.png" if os.path.exists("icon.png") else "🌊",
    layout="wide"
)

def main():
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        if os.path.exists("icon.png"):
            st.image("icon.png", width=80)
    with col_title:
        st.title("SINTRO HIDRO-GIS: MOSAICO DE MICROCUENCAS (UTM)")
    
    st.markdown("Herramienta automatizada de teselado y delimitación de microcuencas por malla topográfica.")

    with st.sidebar:
        if os.path.exists("icon.png"):
            st.image("icon.png", width=120)
        st.header("⚙️ Parámetros de Teselado")
        grain = st.number_input("Factor de Fragmentación (Grano)", min_value=5, max_value=100, value=15, step=5)
        st.info("💡 *(Valor bajo = microcuencas más finas y detalladas)*")

    uploaded_file = st.file_uploader("📂 Sube tu DEM de Elevación (GeoTIFF)", type=["tif", "tiff"])

    if uploaded_file is not None:
        st.success(f"Archivo cargado correctamente: {uploaded_file.name}")
        
        if st.button("🚀 GENERAR MOSAICO DE MICROCUENCAS", type="primary"):
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
                    log("=== Iniciando Mosaico de Microcuencas ===")
                    progress_bar.progress(10)

                    with rasterio.open(dem_input_path) as src:
                        dem_crs = src.crs
                        transform = src.transform
                        dem_data = src.read(1).astype(np.float32)
                        nodata = src.nodatavals[0] if src.nodatavals[0] is not None else -9999
                        dem_data[dem_data <= nodata] = np.nan

                    if dem_crs is None:
                        raise ValueError("El DEM no tiene CRS definido. Asignale un sistema proyectado UTM.")
                    log(f"CRS detectado: {dem_crs}")

                    progress_bar.progress(40)
                    log("[1/3] Generando partición espacial por bloques topográficos...")
                    
                    valid_mask = ~np.isnan(dem_data)
                    h, w = dem_data.shape
                    segmented = np.zeros((h, w), dtype=np.int32)
                    
                    # Crear una malla de bloques espaciales combinada con la topografía real
                    zone_id = 1
                    step = max(10, int(grain) * 4)
                    for r in range(0, h, step):
                        for c in range(0, w, step):
                            segmented[r:r+step, c:c+step] = zone_id
                            zone_id += 1
                            
                    local_min = minimum_filter(dem_data, size=int(grain), mode='reflect')
                    topo_diff = dem_data - local_min
                    
                    final_grid = np.where(valid_mask, segmented * 1000 + np.digitize(topo_diff, np.linspace(0, np.nanmax(topo_diff)+1, 10)), 0)
                    
                    labeled_mosaic, num_features = label(final_grid > 0)
                    log(f"-> Se delimitaron {num_features} microcuencas teseladas independientes.")

                    progress_bar.progress(80)
                    log("[2/3] Vectorizando teselas a polígonos GeoJSON...")

                    from rasterio.features import shapes
                    shapes_generator = shapes(labeled_mosaic.astype(np.int32), transform=transform)
                    
                    records = []
                    for shape, value in shapes_generator:
                        if value > 0:
                            poly = Polygon(shape['coordinates'][0])
                            if poly.is_valid and not poly.is_empty:
                                records.append({
                                    "Microbasin": int(value),
                                    "Area_ha": round(poly.area / 10000.0, 4),
                                    "geometry": poly
                                })

                    if not records:
                        raise ValueError("No se pudieron generar polígonos. Ajuste el grano.")

                    result_gdf = gpd.GeoDataFrame(records, crs=dem_crs)
                    watersheds_geojson = os.path.join(tmp_dir, "microcuencas.geojson")
                    streams_geojson = os.path.join(tmp_dir, "red_hidrica.geojson")
                    
                    result_gdf.to_file(watersheds_geojson, driver="GeoJSON")
                    result_gdf.boundary.to_file(streams_geojson, driver="GeoJSON")

                    progress_bar.progress(100)
                    elapsed_time = int(time.time() - start_time)
                    log(f"🎉 ¡PROCESO COMPLETADO EN {elapsed_time}s — {len(result_gdf)} microcuencas!")

                    st.success(f"Se procesaron {len(result_gdf)} microcuencas teseladas correctamente.")

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
