import os
import time
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import rasterio
import geopandas as gpd

try:
    import whitebox
    WHITEBOX_OK = True
except ImportError:
    WHITEBOX_OK = False


class SyntroWatershedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Syntro GIS - Delimitador Hidrográfico por Umbral (UTM)")
        self.root.geometry("940x800")
        self.root.configure(bg="#0f172a")

        self.dem_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.log_file = None

        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')

        main_frame = tk.Frame(self.root, bg="#1e293b", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        tk.Label(
            main_frame,
            text="SINTRO HIDRO-GIS: MICROCUENCAS + ORDEN DE RÍOS (UTM)",
            font=("Segoe UI", 14, "bold"),
            fg="#38bdf8", bg="#1e293b"
        ).pack(anchor="w", pady=(0, 15))

        self.create_file_selector(
            main_frame, "DEM de Elevación (GeoTIFF):",
            self.dem_path, [("GeoTIFF", "*.tif;*.tiff")]
        )
        self.create_dir_selector(
            main_frame, "Carpeta de Resultados:", self.output_dir
        )

        param_frame = tk.LabelFrame(
            main_frame,
            text=" Parámetros de Delimitación ",
            font=("Segoe UI", 10, "bold"),
            fg="#e2e8f0", bg="#1e293b", padx=10, pady=10
        )
        param_frame.pack(fill=tk.X, pady=10)

        tk.Label(param_frame, text="Umbral de Acumulación (Celdas):",
                 fg="#cbd5e1", bg="#1e293b").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.threshold_entry = tk.Entry(param_frame, width=10, bg="#334155",
                                        fg="#ffffff", insertbackground="white")
        self.threshold_entry.insert(0, "500")
        self.threshold_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(param_frame, text="Puntos de Salida por Río:",
                 fg="#cbd5e1", bg="#1e293b").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.spacing_entry = tk.Entry(param_frame, width=10, bg="#334155",
                                      fg="#ffffff", insertbackground="white")
        self.spacing_entry.insert(0, "200")
        self.spacing_entry.grid(row=0, column=3, padx=5, pady=5)

        tk.Label(param_frame,
                 text="(Umbral BAJO = más microcuencas | Umbral ALTO = menos microcuencas grandes)",
                 fg="#64748b", bg="#1e293b", font=("Segoe UI", 8, "italic")
                 ).grid(row=1, column=0, columnspan=4, sticky="w", padx=5)

        self.run_btn = tk.Button(
            main_frame, text="GENERAR MICROCUENCAS + ORDEN DE RÍOS",
            bg="#0284c7", fg="white",
            font=("Segoe UI", 11, "bold"), relief=tk.FLAT,
            command=self.start_processing_thread
        )
        self.run_btn.pack(fill=tk.X, pady=15)

        progress_frame = tk.Frame(main_frame, bg="#1e293b")
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress = ttk.Progressbar(progress_frame, orient="horizontal",
                                        length=100, mode="determinate")
        self.progress.pack(fill=tk.X, side=tk.TOP, pady=5)

        self.timer_label = tk.Label(progress_frame, text="Tiempo transcurrido: 00:00",
                                    font=("Consolas", 10), fg="#94a3b8", bg="#1e293b")
        self.timer_label.pack(side=tk.LEFT)

        log_frame = tk.LabelFrame(
            main_frame, text=" Registro de Ejecución (Log) ",
            font=("Segoe UI", 10, "bold"),
            fg="#e2e8f0", bg="#1e293b", padx=5, pady=5
        )
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.log_text = tk.Text(log_frame, height=14, bg="#0f172a",
                                fg="#34d399", font=("Consolas", 9),
                                insertbackground="white")
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

    def create_file_selector(self, parent, label_text, string_var, filetypes):
        frame = tk.Frame(parent, bg="#1e293b")
        frame.pack(fill=tk.X, pady=5)
        tk.Label(frame, text=label_text, width=30, anchor="w",
                 fg="#cbd5e1", bg="#1e293b", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        entry = tk.Entry(frame, textvariable=string_var, bg="#334155",
                         fg="#ffffff", insertbackground="white", width=45)
        entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        btn = tk.Button(frame, text="Examinar", bg="#475569", fg="white",
                        relief=tk.FLAT,
                        command=lambda: string_var.set(
                            filedialog.askopenfilename(filetypes=filetypes)))
        btn.pack(side=tk.RIGHT)

    def create_dir_selector(self, parent, label_text, string_var):
        frame = tk.Frame(parent, bg="#1e293b")
        frame.pack(fill=tk.X, pady=5)
        tk.Label(frame, text=label_text, width=30, anchor="w",
                 fg="#cbd5e1", bg="#1e293b", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        entry = tk.Entry(frame, textvariable=string_var, bg="#334155",
                         fg="#ffffff", insertbackground="white", width=45)
        entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        btn = tk.Button(frame, text="Examinar", bg="#475569", fg="white",
                        relief=tk.FLAT,
                        command=lambda: string_var.set(filedialog.askdirectory()))
        btn.pack(side=tk.RIGHT)

    def log(self, message):
        timestamp = time.strftime('%H:%M:%S')
        line = f"[{timestamp}] {message}"
        try:
            self.log_text.insert(tk.END, line + "\n")
            self.log_text.see(tk.END)
        except Exception:
            pass
        if self.log_file:
            try:
                self.log_file.write(line + "\n")
                self.log_file.flush()
            except Exception:
                pass

    def start_processing_thread(self):
        if not WHITEBOX_OK:
            messagebox.showerror("Falta whitebox",
                                 "Instalá WhiteboxTools con:\n\npip install whitebox")
            return

        if not self.dem_path.get() or not self.output_dir.get():
            messagebox.showerror("Error", "Debe seleccionar el DEM y la carpeta de salida.")
            return

        os.makedirs(self.output_dir.get(), exist_ok=True)
        log_path = os.path.join(self.output_dir.get(), "log_ejecucion.txt")
        self.log_file = open(log_path, "w", encoding="utf-8")

        self.run_btn.config(state=tk.DISABLED)
        self.progress['value'] = 0
        threading.Thread(target=self.run_analysis, daemon=True).start()

    # ----------------------------------------------------------------
    # HELPERS PARA FORZAR CRS
    # ----------------------------------------------------------------
    def force_crs_on_raster(self, raster_path, crs):
        """Reescribe un ráster forzando su CRS (por si WhiteboxTools lo dejó sin él)."""
        if not os.path.exists(raster_path):
            return
        with rasterio.open(raster_path) as src:
            if src.crs is not None:
                return  # ya tiene CRS, no tocar
            data = src.read()
            meta = src.meta.copy()
            meta.update({"crs": crs})
        with rasterio.open(raster_path, "w", **meta) as dst:
            dst.write(data)

    def force_crs_on_vector(self, vector_path, crs):
        """Reescribe un shapefile forzando su CRS."""
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

    def extract_stream_outlets(self, streams_path, fdir_path, acc_path, out_path):
        """
        Genera un raster con IDs únicos en las celdas de cabecera de cada
        segmento de río (para generar una microcuenca por segmento).
        """
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

        spacing = int(self.spacing_entry.get())
        sampled_rows = rows[::max(1, spacing)]
        sampled_cols = cols[::max(1, spacing)]

        outlets = np.zeros_like(streams, dtype=np.int32)
        for i, (r, c) in enumerate(zip(sampled_rows, sampled_cols), start=1):
            outlets[r, c] = i

        meta.update({"count": 1, "dtype": "int32", "nodata": 0})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(outlets, 1)

        return len(sampled_rows)

    def run_analysis(self):
        start_time = time.time()
        timer_active = True

        def update_timer():
            while timer_active:
                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                try:
                    self.timer_label.config(text=f"Tiempo transcurrido: {mins:02d}:{secs:02d}")
                except Exception:
                    pass
                time.sleep(1)

        threading.Thread(target=update_timer, daemon=True).start()

        try:
            self.log("=== Microcuencas + Orden de Ríos (CRS UTM) ===")
            self.progress['value'] = 3

            dem_file = self.dem_path.get()
            out_folder = self.output_dir.get()
            threshold = int(self.threshold_entry.get())

            # ---------- CRS del DEM ----------
            with rasterio.open(dem_file) as src:
                dem_crs = src.crs
            if dem_crs is None:
                raise ValueError(
                    "El DEM no tiene CRS definido. Asignale uno (idealmente UTM) "
                    "antes de continuar."
                )
            self.log(f"CRS del DEM: {dem_crs}")
            if not dem_crs.is_projected:
                self.log("ADVERTENCIA: el CRS no es proyectado (no es UTM). "
                         "Las áreas podrían ser incorrectas.")

            wbt = whitebox.WhiteboxTools()
            wbt.set_verbose_mode(False)
            wbt.set_working_dir(out_folder)

            dem_fill         = os.path.join(out_folder, "01_dem_fill.tif")
            dem_fdir         = os.path.join(out_folder, "02_flow_dir.tif")
            dem_acc          = os.path.join(out_folder, "03_flow_accum.tif")
            dem_stream       = os.path.join(out_folder, "04_streams.tif")
            dem_stream_order = os.path.join(out_folder, "05_stream_order.tif")
            dem_outlets      = os.path.join(out_folder, "06_outlets.tif")
            dem_watershed    = os.path.join(out_folder, "07_watersheds.tif")
            streams_vec      = os.path.join(out_folder, "red_hidrica.shp")
            watersheds_vec   = os.path.join(out_folder, "microcuencas.shp")
            streams_geojson  = os.path.join(out_folder, "red_hidrica.geojson")
            watersheds_geojson = os.path.join(out_folder, "microcuencas.geojson")

            # --- 1. Fill
            self.log("[1/8] Rellenando depresiones...")
            self.progress['value'] = 8
            wbt.fill_depressions(dem_file, dem_fill)
            self.force_crs_on_raster(dem_fill, dem_crs)

            # --- 2. Flow Direction
            self.log("[2/8] Calculando dirección de flujo (D8)...")
            self.progress['value'] = 18
            wbt.d8_pointer(dem_fill, dem_fdir)
            self.force_crs_on_raster(dem_fdir, dem_crs)

            # --- 3. Flow Accumulation
            self.log("[3/8] Calculando acumulación de flujo...")
            self.progress['value'] = 30
            wbt.d8_flow_accumulation(dem_fill, dem_acc, out_type='cells')
            self.force_crs_on_raster(dem_acc, dem_crs)

            # --- 4. Streams por umbral
            self.log(f"[4/8] Extrayendo red hídrica con umbral={threshold}...")
            self.progress['value'] = 42
            wbt.extract_streams(dem_acc, dem_stream, threshold=threshold)
            self.force_crs_on_raster(dem_stream, dem_crs)

            with rasterio.open(dem_stream) as src:
                streams_arr = src.read(1)
                n_cells = int((streams_arr > 0).sum())
            self.log(f"  Celdas de río detectadas: {n_cells}")
            if n_cells == 0:
                raise ValueError(f"El umbral {threshold} no generó red. Reducilo.")

            # --- 5. ORDEN DE RÍOS (Strahler)
            self.log("[5/8] Calculando orden de ríos (Strahler)...")
            self.progress['value'] = 52
            try:
                wbt.strahler_stream_order(dem_fdir, dem_stream, dem_stream_order)
                self.force_crs_on_raster(dem_stream_order, dem_crs)
                self.log("  Orden de ríos calculado correctamente.")
            except Exception as e:
                self.log(f"  (Aviso) No se pudo calcular orden Strahler: {e}")
                dem_stream_order = None

            # --- 6. Puntos de salida
            self.log("[6/8] Extrayendo puntos de salida por segmento...")
            self.progress['value'] = 62
            n_outlets = self.extract_stream_outlets(dem_stream, dem_fdir, dem_acc, dem_outlets)
            self.force_crs_on_raster(dem_outlets, dem_crs)
            self.log(f"  Puntos de salida generados: {n_outlets}")

            # --- 7. Watershed
            self.log("[7/8] Delimitando cuenca por cada punto de salida...")
            self.progress['value'] = 78
            wbt.watershed(dem_fdir, dem_outlets, dem_watershed, esri_pntr=False)
            self.force_crs_on_raster(dem_watershed, dem_crs)

            # --- 8. Vectorizar + forzar CRS + unir orden
            self.log("[8/8] Vectorizando resultados y forzando CRS...")
            self.progress['value'] = 90

            # Red hídrica vectorial
            try:
                wbt.raster_streams_to_vector(dem_stream, dem_fdir, streams_vec)
                gdf_s = self.force_crs_on_vector(streams_vec, dem_crs)
            except Exception as e:
                self.log(f"  (Aviso) Vectorización de ríos: {e}")
                gdf_s = None

            # Unir el orden Strahler (ráster) a la red vectorial por ubicación
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
                    self.log(f"  Orden de ríos añadido a la red hídrica.")
                except Exception as e:
                    self.log(f"  (Aviso) No se pudo unir orden a la red: {e}")
                    gdf_s["ORDEN"] = 1
            elif gdf_s is not None:
                gdf_s["ORDEN"] = 1

            # Guardar red hídrica con orden
            if gdf_s is not None:
                try:
                    gdf_s.to_file(streams_vec, driver="ESRI Shapefile")
                    gdf_s.to_file(streams_geojson, driver="GeoJSON")
                    self.log(f"  Red hídrica: {streams_geojson} ({len(gdf_s)} segmentos)")
                except Exception as e:
                    self.log(f"  (Aviso) Guardado de red: {e}")

            # Microcuencas
            try:
                wbt.raster_to_vector_polygons(dem_watershed, watersheds_vec)
                gdf_w = self.force_crs_on_vector(watersheds_vec, dem_crs)
                if gdf_w is not None:
                    gdf_w["Area_ha"] = gdf_w.geometry.area / 10000.0
                    gdf_w.to_file(watersheds_geojson, driver="GeoJSON")
                    self.log(f"  Microcuencas: {watersheds_geojson} ({len(gdf_w)} cuencas)")
            except Exception as e:
                self.log(f"  (Aviso) Microcuencas: {e}")

            self.progress['value'] = 100
            self.log("=" * 50)
            self.log(f"PROCESO COMPLETADO — {n_outlets} microcuencas")
            self.log(f"CRS aplicado: {dem_crs}")
            self.log(f"Carpeta: {out_folder}")
            self.log("=" * 50)

            messagebox.showinfo(
                "Proceso Exitoso",
                f"Se generaron {n_outlets} microcuencas y la red hídrica con orden.\n\n"
                f"CRS: {dem_crs}\n"
                f"GeoJSON: {watersheds_geojson}"
            )

        except Exception as e:
            tb = traceback.format_exc()
            self.log(f"ERROR: {str(e)}")
            for line in tb.splitlines():
                self.log(f"  {line}")
            try:
                messagebox.showerror("Error", str(e))
            except Exception:
                pass
        finally:
            timer_active = False
            try:
                self.run_btn.config(state=tk.NORMAL)
            except Exception:
                pass
            if self.log_file:
                try:
                    self.log_file.close()
                except Exception:
                    pass


if __name__ == "__main__":
    if not WHITEBOX_OK:
        print("ADVERTENCIA: instalar whitebox con:  pip install whitebox")
    root = tk.Tk()
    app = SyntroWatershedApp(root)
    root.mainloop()
