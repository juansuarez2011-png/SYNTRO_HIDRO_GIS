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
    page_title="Syntro GIS - Delimitador Hidrográfico",
    page_icon="icon.png" if os.path.exists("icon.png") else "🌊",
    layout="wide"
)

def force_crs_on_vector(vector_path, crs):
    if not os.path.exists(vector_path):
        return None
    gdf = gpd.read_file(vector_path)
    gdf = gdf.set_crs(crs, allow_override=True)
    try:
        gdf.to_file(vector_path, driver="GeoJSON")
    except Exception:
        pass
    return gdf

def main():
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        if os.path.exists("icon.png"):
            st.image("icon.png", width=80)
    with col_title:
        st.title("SINTRO HIDRO-GIS: Microcuencas y Mesa de Drenaje (UTM)")
    
    st.markdown("Herramienta automatizada de análisis hidrológico optimizada para Streamlit Cloud.")

    with st.sidebar:
        if os.path.exists("icon.png"):
            st.image("icon.png", width=120)
        st.header("⚙️ Parámetros de Análisis")
        threshold = st.number_input("Umbral de Fragmentación (Grano)", min_value=5, max_value=200, value=15, step=5)
        st.info("💡 **Consejo:** Un valor bajo genera más microcuencas teseladas detalladas.")

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
                    log("=== Iniciando Pipeline Hidrológico Matricial ===")
                    progress_bar.progress(10)

                    with rasterio.open(dem_input_path) as src:
                        dem_crs = src.crs
                        transform = src.transform
                        dem_data = src.read(1).astype(np.float32)
                        nodata = src.nodatavals[0] if src.nodatavals[0] is not None else -9999
                        dem_data[dem_data <= nodata] = np.nan

                    if dem_crs is None:
                        raise ValueError("El DEM no tiene CRS definido. Asignale un sistema proyectado UTM.")
                    log(f"CRS detectado en el DEM: {dem_crs}")

                    progress_bar.progress(30)
                    log("[1/3] Rellenando depresiones y modelando gradientes topográficos...")
                    
                    valid_mask = ~np.isnan(dem_data)
                    filled = minimum_filter(dem_data, size=int(threshold), mode='reflect')
                    gradient_grid = dem_data - filled

                    progress_bar.progress(60)
                    log("[2/3] Generando partición teselada de microcuencas...")
                    
                    bins = np.nanpercentile(gradient_grid[valid_mask], np.linspace(1, 99, 40))
                    segmented_matrix = np.digitize(gradient_grid, bins)
                    segmented_matrix[~valid_mask] = 0
                    
                    labeled_mosaic, num_features = label(segmented_matrix > 0)
                    log(f"-> Se delimitaron {num_features} microcuencas teseladas conectadas.")

                    progress_bar.progress(85)
                    log("[3/3] Vectorizando teselas a formato GeoJSON...")

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
                        raise ValueError("No se pudieron generar polígonos válidos. Ajuste el umbral.")

                    result_gdf = gpd.GeoDataFrame(records, crs=dem_crs)
                    watersheds_geojson = os.path.join(tmp_dir, "microcuencas.geojson")
                    result_gdf.to_file(watersheds_geojson, driver="GeoJSON")

                    progress_bar.progress(100)
                    elapsed_time = int(time.time() - start_time)
                    log(f"🎉 ¡PROCESO COMPLETADO EN {elapsed_time}s — {len(result_gdf)} microcuencas generadas!")

                    st.success(f"Se procesaron {len(result_gdf)} microcuencas correctamente.")

                    if os.path.exists(watersheds_geojson):
                        with open(watersheds_geojson, "rb") as f:
                            st.download_button(
                                label="📥 Descargar Microcuencas (GeoJSON)",
                                data=f,
                                file_name="microcuencas.geojson",
                                mime="application/geo+json"
                            )

                except Exception as e:
                    st.error(f"Error durante el procesamiento: {str(e)}")
                    st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
