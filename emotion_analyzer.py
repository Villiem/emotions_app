#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 23:49:45 2026

@author: emiliano

Emotion Analyzer - Aplicación para análisis de emociones en video
con ponderación por potencia (Power Law)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import polars as pl
import plotly.graph_objects as go
import numpy as np
import webbrowser
from pathlib import Path


class EmotionAnalyzerApp:
    
    # Columnas originales en inglés (para leer el archivo)
    EMOTION_COLS = ["Negative", "Disgust", "Fear", "Sadness", "Skepticism", 
                    "Neutral", "Surprise", "Delight"]
    
    # Traducción al español (Sadness → Atención/Tristeza)
    EMOTION_TRANSLATION = {
        "Negative": "Negativo",
        "Disgust": "Disgusto", 
        "Fear": "Miedo",
        "Sadness": "Atención/Tristeza",
        "Skepticism": "Escepticismo",
        "Neutral": "Neutral",
        "Surprise": "Sorpresa",
        "Delight": "Deleite"
    }
    
    # Multiplicadores por emoción (2.5 para todas excepto Atención/Tristeza que es 2.0)
    EMOTION_MULTIPLIERS = {
        "Negativo": 2.5,
        "Disgusto": 2.5,
        "Miedo": 2.5,
        "Atención/Tristeza": 2.0,
        "Escepticismo": 2.5,
        "Neutral": 2.5,
        "Sorpresa": 2.5,
        "Deleite": 2.5
    }
    
    # Paletas de colores predefinidas
    PALETTES = {
        "Default": {
            "Negativo": "#DE2000",
            "Disgusto": "#F7FF4D",
            "Miedo": "#7A7A7A",
            "Atención/Tristeza": "#3D8CE3",
            "Escepticismo": "#9304CC",
            "Neutral": "#95A5A6",
            "Sorpresa": "#05CC04",
            "Deleite": "#F5BB27"
        },
        "Vibrante": {
            "Negativo": "#E74C3C",
            "Disgusto": "#9B59B6",
            "Miedo": "#34495E",
            "Atención/Tristeza": "#3498DB",
            "Escepticismo": "#F39C12",
            "Neutral": "#95A5A6",
            "Sorpresa": "#1ABC9C",
            "Deleite": "#2ECC71"
        },
        "Escala de grises": {
            "Negativo": "#1A1A1A",
            "Disgusto": "#333333",
            "Miedo": "#4D4D4D",
            "Atención/Tristeza": "#666666",
            "Escepticismo": "#808080",
            "Neutral": "#999999",
            "Sorpresa": "#B3B3B3",
            "Deleite": "#CCCCCC"
        },
        "Pastel": {
            "Negativo": "#FFB3BA",
            "Disgusto": "#BAFFC9",
            "Miedo": "#BAE1FF",
            "Atención/Tristeza": "#FFFFBA",
            "Escepticismo": "#FFD9BA",
            "Neutral": "#E0BBE4",
            "Sorpresa": "#D4F0F0",
            "Deleite": "#CCE2CB"
        },
        "Cálidos": {
            "Negativo": "#D32F2F",
            "Disgusto": "#F57C00",
            "Miedo": "#FFA000",
            "Atención/Tristeza": "#FFCA28",
            "Escepticismo": "#FFE082",
            "Neutral": "#A1887F",
            "Sorpresa": "#FF7043",
            "Deleite": "#FF5722"
        },
        "Fríos": {
            "Negativo": "#0D47A1",
            "Disgusto": "#1565C0",
            "Miedo": "#1976D2",
            "Atención/Tristeza": "#2196F3",
            "Escepticismo": "#42A5F5",
            "Neutral": "#64B5F6",
            "Sorpresa": "#00ACC1",
            "Deleite": "#26A69A"
        },
        "Alto contraste": {
            "Negativo": "#FF0000",
            "Disgusto": "#00FF00",
            "Miedo": "#0000FF",
            "Atención/Tristeza": "#FFFF00",
            "Escepticismo": "#FF00FF",
            "Neutral": "#00FFFF",
            "Sorpresa": "#FF8000",
            "Deleite": "#8000FF"
        }
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Analyzer")
        self.root.geometry("750x700")
        self.root.resizable(True, True)
        
        # Variables
        self.filepath = tk.StringVar()
        self.a_param = tk.DoubleVar(value=4.5167)
        self.b_param = tk.DoubleVar(value=-0.228)
        self.remove_last_second = tk.BooleanVar(value=False)
        self.selected_palette = tk.StringVar(value="Default")
        self.df_original = None
        self.df_weighted = None
        
        # Configuración individual por emoción: {emocion: {"color": str, "width": IntVar, "visible": BooleanVar}}
        self.emotion_config = {}
        for emotion in self.PALETTES["Default"].keys():
            self.emotion_config[emotion] = {
                "color": self.PALETTES["Default"][emotion],
                "width": tk.IntVar(value=2),
                "visible": tk.BooleanVar(value=True)
            }
        
        self.setup_ui()
    
    def setup_ui(self):
        # Frame principal con padding
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Sección: Selección de archivo ===
        file_frame = ttk.LabelFrame(main_frame, text="📁 Archivo de datos", padding="8")
        file_frame.pack(fill=tk.X, pady=(0, 8))
        
        ttk.Entry(file_frame, textvariable=self.filepath, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(file_frame, text="Seleccionar...", command=self.select_file).pack(side=tk.RIGHT)
        
        # === Sección: Parámetros ===
        params_frame = ttk.LabelFrame(main_frame, text="⚙️ Parámetros de ponderación (y = a × t^b)", padding="8")
        params_frame.pack(fill=tk.X, pady=(0, 8))
        
        params_inner = ttk.Frame(params_frame)
        params_inner.pack(fill=tk.X)
        
        ttk.Label(params_inner, text="a:").pack(side=tk.LEFT)
        ttk.Entry(params_inner, textvariable=self.a_param, width=8).pack(side=tk.LEFT, padx=(5, 15))
        ttk.Label(params_inner, text="b:").pack(side=tk.LEFT)
        ttk.Entry(params_inner, textvariable=self.b_param, width=8).pack(side=tk.LEFT, padx=(5, 15))
        ttk.Checkbutton(params_inner, text="Quitar último segundo", 
                        variable=self.remove_last_second).pack(side=tk.LEFT, padx=15)
        
        # === Sección: Personalización de emociones ===
        style_frame = ttk.LabelFrame(main_frame, text="🎨 Personalización de emociones", padding="8")
        style_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
        # Selector de paleta
        palette_frame = ttk.Frame(style_frame)
        palette_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(palette_frame, text="Paleta:").pack(side=tk.LEFT)
        palette_combo = ttk.Combobox(palette_frame, textvariable=self.selected_palette, 
                                      values=list(self.PALETTES.keys()), state="readonly", width=20)
        palette_combo.pack(side=tk.LEFT, padx=10)
        palette_combo.bind("<<ComboboxSelected>>", self.apply_palette)
        
        ttk.Button(palette_frame, text="Aplicar paleta", command=self.apply_palette).pack(side=tk.LEFT, padx=5)
        
        # Grid de emociones
        emotions_grid = ttk.Frame(style_frame)
        emotions_grid.pack(fill=tk.BOTH, expand=True)
        
        # Headers
        ttk.Label(emotions_grid, text="Visible", font=("Arial", 9, "bold")).grid(row=0, column=0, padx=5, pady=3)
        ttk.Label(emotions_grid, text="Emoción", font=("Arial", 9, "bold")).grid(row=0, column=1, padx=5, pady=3, sticky="w")
        ttk.Label(emotions_grid, text="Color", font=("Arial", 9, "bold")).grid(row=0, column=2, padx=5, pady=3)
        ttk.Label(emotions_grid, text="Grosor", font=("Arial", 9, "bold")).grid(row=0, column=3, padx=5, pady=3)
        
        self.color_buttons = {}
        
        for i, (emotion, config) in enumerate(self.emotion_config.items(), start=1):
            # Checkbox visible
            ttk.Checkbutton(emotions_grid, variable=config["visible"]).grid(row=i, column=0, padx=5, pady=2)
            
            # Nombre emoción
            ttk.Label(emotions_grid, text=emotion, width=12).grid(row=i, column=1, padx=5, pady=2, sticky="w")
            
            # Botón color
            color_btn = tk.Button(emotions_grid, bg=config["color"], width=6, height=1,
                                  relief=tk.FLAT, cursor="hand2",
                                  command=lambda e=emotion: self.pick_color(e))
            color_btn.grid(row=i, column=2, padx=5, pady=2)
            self.color_buttons[emotion] = color_btn
            
            # Spinbox grosor
            ttk.Spinbox(emotions_grid, from_=1, to=6, textvariable=config["width"], 
                        width=5).grid(row=i, column=3, padx=5, pady=2)
        
        # Botones rápidos
        quick_frame = ttk.Frame(style_frame)
        quick_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(quick_frame, text="Mostrar todas", command=self.show_all_emotions).pack(side=tk.LEFT, padx=3)
        ttk.Button(quick_frame, text="Ocultar todas", command=self.hide_all_emotions).pack(side=tk.LEFT, padx=3)
        ttk.Button(quick_frame, text="Grosor +", command=lambda: self.adjust_all_widths(1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(quick_frame, text="Grosor -", command=lambda: self.adjust_all_widths(-1)).pack(side=tk.LEFT, padx=3)
        
        # === Sección: Acciones ===
        actions_frame = ttk.LabelFrame(main_frame, text="🚀 Acciones", padding="8")
        actions_frame.pack(fill=tk.X, pady=(0, 8))
        
        btn_frame = ttk.Frame(actions_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="📊 Procesar datos", command=self.process_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📈 Ver gráfica", command=self.show_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📊 Positivo vs Neutral", command=self.show_bar_chart).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="💾 Exportar todo", command=self.export_all).pack(side=tk.LEFT, padx=5)
        
        # === Sección: Log/Status ===
        log_frame = ttk.LabelFrame(main_frame, text="📋 Estado", padding="8")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=5, state=tk.DISABLED, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log("Aplicación iniciada. Selecciona un archivo Excel para comenzar.")
    
    def apply_palette(self, event=None):
        """Aplica la paleta seleccionada."""
        palette_name = self.selected_palette.get()
        palette = self.PALETTES.get(palette_name, self.PALETTES["Vibrante"])
        
        for emotion, color in palette.items():
            if emotion in self.emotion_config:
                self.emotion_config[emotion]["color"] = color
                self.color_buttons[emotion].configure(bg=color)
        
        self.log(f"Paleta '{palette_name}' aplicada")
    
    def pick_color(self, emotion):
        """Abre selector de color para una emoción."""
        current_color = self.emotion_config[emotion]["color"]
        color = colorchooser.askcolor(initialcolor=current_color, title=f"Color para {emotion}")
        
        if color[1]:
            self.emotion_config[emotion]["color"] = color[1]
            self.color_buttons[emotion].configure(bg=color[1])
            self.log(f"Color de {emotion}: {color[1]}")
    
    def show_all_emotions(self):
        for config in self.emotion_config.values():
            config["visible"].set(True)
    
    def hide_all_emotions(self):
        for config in self.emotion_config.values():
            config["visible"].set(False)
    
    def adjust_all_widths(self, delta):
        for config in self.emotion_config.values():
            new_val = config["width"].get() + delta
            config["width"].set(max(1, min(6, new_val)))
    
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
            self.log(f"Archivo: {Path(filepath).name}")
    
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
        
        # Renombrar columnas al español
        rename_dict = {eng: self.EMOTION_TRANSLATION[eng] 
                       for eng in available_emotions 
                       if eng in self.EMOTION_TRANSLATION}
        df_t = df_t.rename(rename_dict)
        
        spanish_emotions = [self.EMOTION_TRANSLATION.get(e, e) for e in available_emotions]
        
        return df_t.select(["segundos"] + spanish_emotions)
    
    def power_weights(self, t, a, b, normalize=False):
        """Calcula pesos con función potencia."""
        t = np.array(t, dtype=float)
        t = np.where(t == 0, 0.5, t)
        weights = a * np.power(t, b)
        if normalize:
            weights = weights / weights.sum()
        return weights
    
    def apply_power_weighting(self, df, a, b):
        """Aplica ponderación por potencia y multiplicadores por emoción."""
        segundos = df["segundos"].to_numpy()
        weights = self.power_weights(segundos, a, b)
        emotion_cols = [col for col in df.columns if col != "segundos"]
        
        # Aplicar peso temporal Y multiplicador por emoción
        weighted_cols = []
        for col in emotion_cols:
            multiplier = self.EMOTION_MULTIPLIERS.get(col, 1.0)
            weighted_cols.append(
                (pl.col(col) * pl.lit(weights) * multiplier).alias(f"{col}_pw")
            )
        
        return df.select(
            pl.col("segundos"),
            pl.lit(weights).alias("peso_temporal"),
            *weighted_cols
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
            self.log(f"Ponderación aplicada (a={a}, b={b})")
            self.df_weighted = self.apply_power_weighting(self.df_original, a, b)
            
            n_rows = len(self.df_original)
            n_emotions = len([c for c in self.df_original.columns if c != "segundos"])
            self.log(f"✅ {n_rows} segundos, {n_emotions} emociones")
            
            messagebox.showinfo("Éxito", f"Datos procesados.\n{n_rows} puntos, {n_emotions} emociones.")
            
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", f"Error al procesar:\n{str(e)}")
    
    def create_bar_chart(self):
        """Crea gráfica de barras apiladas: Positivo (Deleite) vs Neutral (suma del resto)."""
        if self.df_weighted is None:
            return None
        
        df = self.df_weighted
        
        # Filtrar último segundo si está activado
        if self.remove_last_second.get():
            max_sec = df["segundos"].max()
            df = df.filter(pl.col("segundos") < max_sec)
        
        segundos = df["segundos"].to_list()
        
        # Positivo = Deleite
        positivo = df["Deleite_pw"].to_list() if "Deleite_pw" in df.columns else [0] * len(segundos)
        
        # Neutral = suma de las demás emociones (con nuevo nombre Atención/Tristeza)
        neutral_cols = ["Negativo_pw", "Disgusto_pw", "Miedo_pw", "Atención/Tristeza_pw", 
                        "Escepticismo_pw", "Sorpresa_pw"]
        available_cols = [c for c in neutral_cols if c in df.columns]
        
        if available_cols:
            neutral = df.select(pl.sum_horizontal(available_cols)).to_series().to_list()
        else:
            neutral = [0] * len(segundos)
        
        fig = go.Figure()
        
        # Barras Positivo (Deleite) - abajo
        fig.add_trace(go.Bar(
            x=segundos,
            y=positivo,
            name="Positivo",
            marker_color="#1565C0"
        ))
        
        # Barras Neutral (suma) - arriba
        fig.add_trace(go.Bar(
            x=segundos,
            y=neutral,
            name="Neutral",
            marker_color="#E65100"
        ))
        
        fig.update_layout(
            title="Positivo vs Neutral (Ponderado)",
            xaxis_title="Segundos",
            yaxis_title="Intensidad",
            barmode="stack",  # Barras apiladas verticalmente
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        
        return fig
    
    def create_plot(self):
        """Crea la gráfica de emociones."""
        if self.df_original is None:
            return None
        
        fig = go.Figure()
        emotion_cols = [c for c in self.df_original.columns if c != "segundos"]
        
        # Filtrar último segundo si está activado
        df_orig = self.df_original
        df_weight = self.df_weighted
        
        if self.remove_last_second.get():
            max_sec = df_orig["segundos"].max()
            df_orig = df_orig.filter(pl.col("segundos") < max_sec)
            df_weight = df_weight.filter(pl.col("segundos") < max_sec)
        
        for col in emotion_cols:
            config = self.emotion_config.get(col)
            if not config or not config["visible"].get():
                continue
            
            color = config["color"]
            width = config["width"].get()
            
            # Original (punteado)
            fig.add_trace(go.Scatter(
                x=df_orig["segundos"], 
                y=df_orig[col],
                name=col, 
                mode='lines', 
                line=dict(width=max(1, width-1), dash='dot', color=color),
                legendgroup=col
            ))
            # Ponderado (sólido)
            fig.add_trace(go.Scatter(
                x=df_weight["segundos"], 
                y=df_weight[f"{col}_pw"],
                name=f"{col} (Pond.)", 
                mode='lines', 
                line=dict(width=width, color=color),
                legendgroup=col
            ))
        
        fig.update_layout(
            title="Emociones - Original (punteado) vs Ponderado (sólido)",
            xaxis_title="Segundos",
            yaxis_title="Intensidad",
            legend_title="Emociones",
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12),
            legend=dict(groupclick="toggleitem")
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
            self.log("Gráfica abierta en navegador")
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
    
    def show_bar_chart(self):
        """Muestra la gráfica de barras Positivo vs Neutral."""
        if self.df_weighted is None:
            messagebox.showwarning("Aviso", "Procesa los datos primero.")
            return
        
        try:
            fig = self.create_bar_chart()
            temp_path = Path.home() / "emotion_bar_chart_temp.html"
            fig.write_html(str(temp_path))
            webbrowser.open(f"file://{temp_path}")
            self.log("Gráfica de barras abierta en navegador")
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
    
    def export_all(self):
        """Exporta datos procesados y gráfica."""
        if self.df_original is None:
            messagebox.showwarning("Aviso", "Procesa los datos primero.")
            return
        
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
            
            # Exportar gráfica de barras
            bar_path = export_path / f"{base_name}_positivo_vs_neutral.html"
            fig_bar = self.create_bar_chart()
            fig_bar.write_html(str(bar_path))
            self.log(f"Exportado: {bar_path.name}")
            
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
            self.log(f"❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    app = EmotionAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
