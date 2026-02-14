#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emotion Analyzer - Aplicación para análisis de emociones en video
con ponderación por potencia (Power Law)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import polars as pl
import plotly.graph_objects as go
import numpy as np
import os
import webbrowser
from pathlib import Path


class EmotionAnalyzerApp:
    
    EMOTION_COLS = ["Negative", "Disgust", "Fear", "Sadness", "Skepticism", 
                    "Neutral", "Surprise", "Delight"]
    
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Analyzer")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Variables
        self.filepath = tk.StringVar()
        self.a_param = tk.DoubleVar(value=4.5167)
        self.b_param = tk.DoubleVar(value=-0.228)
        self.df_original = None
        self.df_weighted = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Frame principal con padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Sección: Selección de archivo ===
        file_frame = ttk.LabelFrame(main_frame, text="📁 Archivo de datos", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Entry(file_frame, textvariable=self.filepath, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(file_frame, text="Seleccionar...", command=self.select_file).pack(side=tk.RIGHT)
        
        # === Sección: Parámetros ===
        params_frame = ttk.LabelFrame(main_frame, text="⚙️ Parámetros de ponderación (y = a × t^b)", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Parámetro a
        a_frame = ttk.Frame(params_frame)
        a_frame.pack(fill=tk.X, pady=5)
        ttk.Label(a_frame, text="a (escala):").pack(side=tk.LEFT)
        ttk.Entry(a_frame, textvariable=self.a_param, width=10).pack(side=tk.LEFT, padx=10)
        ttk.Label(a_frame, text="(default: 4.5167)", foreground="gray").pack(side=tk.LEFT)
        
        # Parámetro b
        b_frame = ttk.Frame(params_frame)
        b_frame.pack(fill=tk.X, pady=5)
        ttk.Label(b_frame, text="b (decaimiento):").pack(side=tk.LEFT)
        ttk.Entry(b_frame, textvariable=self.b_param, width=10).pack(side=tk.LEFT, padx=10)
        ttk.Label(b_frame, text="(más negativo = decae más rápido)", foreground="gray").pack(side=tk.LEFT)
        
        # === Sección: Acciones ===
        actions_frame = ttk.LabelFrame(main_frame, text="🚀 Acciones", padding="10")
        actions_frame.pack(fill=tk.X, pady=(0, 15))
        
        btn_frame = ttk.Frame(actions_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="📊 Procesar datos", command=self.process_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📈 Ver gráfica", command=self.show_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="💾 Exportar todo", command=self.export_all).pack(side=tk.LEFT, padx=5)
        
        # === Sección: Log/Status ===
        log_frame = ttk.LabelFrame(main_frame, text="📋 Estado", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=10, state=tk.DISABLED, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log("Aplicación iniciada. Selecciona un archivo Excel para comenzar.")
    
    def log(self, message):
        """Agrega mensaje al log."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"• {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
    
    def select_file(self):
        """Abre diálogo para seleccionar archivo Excel."""
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo de emociones",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filepath:
            self.filepath.set(filepath)
            self.log(f"Archivo seleccionado: {Path(filepath).name}")
    
    def preprocess_data(self, df):
        """Detecta el formato del df y lo normaliza."""
        first_col = df.columns[0]
        has_interval_cols = any("s-" in str(c) for c in df.columns)
        
        if has_interval_cols:
            cols_to_drop = [c for c in ["E AVG", "E MAX"] if c in df.columns]
            if cols_to_drop:
                df = df.drop(cols_to_drop)
            df = df.slice(1)
            time_cols = [c for c in df.columns if "s-" in c]
            segundos = [int(c.split("-")[1].rstrip("s")) for c in time_cols]
        else:
            cols_to_drop = [c for c in df.columns if "UNNAMED" in str(c) and c != first_col]
            if cols_to_drop:
                df = df.drop(cols_to_drop)
            time_cols = [c for c in df.columns if c != first_col]
            segundos = [int(c) for c in time_cols]
        
        emotion_names = df[first_col].to_list()
        df_t = df.select(time_cols).transpose()
        df_t = df_t.rename({f"column_{i}": name for i, name in enumerate(emotion_names)})
        df_t = df_t.with_columns(pl.Series("segundos", segundos))
        
        available_emotions = [e for e in self.EMOTION_COLS if e in df_t.columns]
        df_t = df_t.with_columns([
            pl.col("segundos").cast(pl.Int16),
            *[pl.col(e).cast(pl.Float64) for e in available_emotions]
        ])
        
        return df_t.select(["segundos"] + available_emotions)
    
    def power_weights(self, t, a, b, normalize=False):
        """Calcula pesos con función potencia."""
        t = np.array(t, dtype=float)
        t = np.where(t == 0, 0.5, t)
        weights = a * np.power(t, b)
        if normalize:
            weights = weights / weights.sum()
        return weights
    
    def apply_power_weighting(self, df, a, b):
        """Aplica ponderación por potencia."""
        segundos = df["segundos"].to_numpy()
        weights = self.power_weights(segundos, a, b)
        emotion_cols = [col for col in df.columns if col != "segundos"]
        
        return df.select(
            pl.col("segundos"),
            pl.lit(weights).alias("peso_temporal"),
            *[(pl.col(col) * pl.lit(weights)).alias(f"{col}_pw") for col in emotion_cols]
        )
    
    def process_data(self):
        """Procesa el archivo seleccionado."""
        if not self.filepath.get():
            messagebox.showwarning("Aviso", "Selecciona un archivo primero.")
            return
        
        try:
            self.log("Cargando archivo...")
            raw_df = pl.read_excel(self.filepath.get())
            
            self.log("Preprocesando datos...")
            self.df_original = self.preprocess_data(raw_df)
            
            a = self.a_param.get()
            b = self.b_param.get()
            self.log(f"Aplicando ponderación (a={a}, b={b})...")
            self.df_weighted = self.apply_power_weighting(self.df_original, a, b)
            
            n_rows = len(self.df_original)
            n_emotions = len([c for c in self.df_original.columns if c != "segundos"])
            self.log(f"✅ Procesado: {n_rows} segundos, {n_emotions} emociones")
            
            messagebox.showinfo("Éxito", f"Datos procesados correctamente.\n{n_rows} puntos temporales, {n_emotions} emociones.")
            
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Error al procesar:\n{str(e)}")
    
    def create_plot(self):
        """Crea la gráfica de emociones."""
        if self.df_original is None:
            return None
        
        fig = go.Figure()
        emotion_cols = [c for c in self.df_original.columns if c != "segundos"]
        
        for col in emotion_cols:
            # Original (punteado)
            fig.add_trace(go.Scatter(
                x=self.df_original["segundos"], 
                y=self.df_original[col],
                name=col, 
                mode='lines', 
                line=dict(width=1, dash='dot')
            ))
            # Ponderado (sólido)
            fig.add_trace(go.Scatter(
                x=self.df_weighted["segundos"], 
                y=self.df_weighted[f"{col}_pw"],
                name=f"{col} (Ponderado)", 
                mode='lines', 
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Emociones - Original vs Ponderación Potencia",
            xaxis_title="Segundos",
            yaxis_title="Intensidad",
            legend_title="Emociones",
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        return fig
    
    def show_plot(self):
        """Muestra la gráfica en el navegador."""
        if self.df_original is None:
            messagebox.showwarning("Aviso", "Procesa los datos primero.")
            return
        
        try:
            fig = self.create_plot()
            temp_path = Path.home() / "emotion_plot_temp.html"
            fig.write_html(str(temp_path))
            webbrowser.open(f"file://{temp_path}")
            self.log(f"Gráfica abierta en navegador")
        except Exception as e:
            self.log(f"❌ Error al mostrar gráfica: {str(e)}")
            messagebox.showerror("Error", str(e))
    
    def export_all(self):
        """Exporta datos procesados y gráfica."""
        if self.df_original is None:
            messagebox.showwarning("Aviso", "Procesa los datos primero.")
            return
        
        # Seleccionar carpeta de destino
        export_dir = filedialog.askdirectory(title="Seleccionar carpeta de exportación")
        if not export_dir:
            return
        
        try:
            export_path = Path(export_dir)
            base_name = Path(self.filepath.get()).stem
            
            # Exportar datos originales
            original_path = export_path / f"{base_name}_original.xlsx"
            self.df_original.write_excel(str(original_path))
            self.log(f"Exportado: {original_path.name}")
            
            # Exportar datos ponderados
            weighted_path = export_path / f"{base_name}_ponderado.xlsx"
            self.df_weighted.write_excel(str(weighted_path))
            self.log(f"Exportado: {weighted_path.name}")
            
            # Exportar gráfica
            plot_path = export_path / f"{base_name}_grafica.html"
            fig = self.create_plot()
            fig.write_html(str(plot_path))
            self.log(f"Exportado: {plot_path.name}")
            
            # Exportar curva de pesos
            curve_path = export_path / f"{base_name}_curva_pesos.html"
            t = self.df_original["segundos"].to_numpy()
            weights = self.power_weights(t, self.a_param.get(), self.b_param.get())
            
            fig_curve = go.Figure()
            fig_curve.add_trace(go.Scatter(
                x=t, y=weights,
                mode='lines+markers',
                name=f'y = {self.a_param.get():.4f} × t^({self.b_param.get():.3f})',
                line=dict(width=2, color='red')
            ))
            fig_curve.update_layout(
                title="Curva de Ponderación Temporal",
                xaxis_title="Segundos",
                yaxis_title="Peso",
                template="plotly_white"
            )
            fig_curve.write_html(str(curve_path))
            self.log(f"Exportado: {curve_path.name}")
            
            self.log("✅ Exportación completa!")
            messagebox.showinfo("Éxito", f"Archivos exportados en:\n{export_dir}")
            
        except Exception as e:
            self.log(f"❌ Error al exportar: {str(e)}")
            messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    app = EmotionAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
