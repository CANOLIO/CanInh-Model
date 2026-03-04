"""
audit_dataset.py
================
Script de auditoría profunda para el dataset de Inhibidores de Cáncer.
Escanea de forma eficiente (sin desbordar la memoria RAM) todos los archivos 
.h5 y .csv para revelar su estructura, dimensiones y distribución de etiquetas.
"""

import os
import glob
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajusta esta ruta si es necesario
DATA_DIR = Path('/Users/fabian/Downloads/CANCERDATASET')
OUTPUT_FILE = Path('./dataset_audit_report.txt')

def count_lines_efficiently(filepath):
    """Cuenta las líneas de un archivo grande sin cargarlo en memoria."""
    def _make_gen(reader):
        b = reader(1024 * 1024)
        while b:
            yield b
            b = reader(1024 * 1024)
    with open(filepath, 'rb') as f:
        count = sum(buf.count(b'\n') for buf in _make_gen(f.raw.read))
    return count

def audit_h5_file(filepath, f_out):
    """Audita la estructura interna de un archivo HDF5."""
    f_out.write(f"\\n{'='*60}\\n")
    f_out.write(f" 📦 ARCHIVO HDF5 : {filepath.name}\\n")
    f_out.write(f" Tamaño       : {filepath.stat().st_size / (1024*1024):.2f} MB\\n")
    f_out.write(f"{'='*60}\\n")
    
    try:
        with h5py.File(filepath, 'r') as hf:
            f_out.write(" Estructura interna:\\n")
            
            # Función recursiva para explorar grupos y datasets
            def explore_node(name, node):
                indent = "   " * (name.count('/') + 1)
                if isinstance(node, h5py.Dataset):
                    dtype = node.dtype
                    shape = node.shape
                    f_out.write(f"{indent}📄 Dataset: '{name}' | Shape: {shape} | Dtype: {dtype}\\n")
                    
                    # Si es el label, veamos la distribución
                    if name == 'label':
                        labels = node[()]
                        n_pos = np.sum(labels == 1)
                        n_neg = np.sum(labels == 0)
                        f_out.write(f"{indent}   ↳ Distribución: {n_pos} Inhibidores (1), {n_neg} No-Inhibidores (0)\\n")
                elif isinstance(node, h5py.Group):
                    f_out.write(f"{indent}📁 Grupo: '{name}'\\n")
            
            hf.visititems(explore_node)
            
    except Exception as e:
        f_out.write(f" ❌ Error al leer HDF5: {e}\\n")

def audit_csv_file(filepath, f_out):
    """Audita un archivo CSV leyendo solo las primeras filas para ahorrar RAM."""
    f_out.write(f"\\n{'='*60}\\n")
    f_out.write(f" 📊 ARCHIVO CSV  : {filepath.name}\\n")
    f_out.write(f" Tamaño       : {filepath.stat().st_size / (1024*1024):.2f} MB\\n")
    f_out.write(f"{'='*60}\\n")
    
    try:
        # Contar filas totales eficientemente
        total_rows = count_lines_efficiently(filepath)
        
        # Leer solo 5 filas para inferir estructura
        df_head = pd.read_csv(filepath, nrows=5, header=None)
        n_cols = df_head.shape[1]
        
        f_out.write(f" Filas totales estimadas : {total_rows}\\n")
        f_out.write(f" Columnas detectadas     : {n_cols}\\n")
        
        # Inferir qué es la primera columna
        first_col_vals = df_head.iloc[:, 0].unique()
        if set(first_col_vals).issubset({0, 1, 0.0, 1.0}):
            f_out.write(f" ↳ La columna 0 parece ser el 'Label' (valores: {first_col_vals}).\\n")
        else:
            f_out.write(f" ↳ La columna 0 tiene valores como: {first_col_vals[:3]}\\n")
            
        f_out.write(f" ↳ Tipos de datos en columnas [0:5]: {df_head.dtypes.iloc[0:5].tolist()}\\n")
        
    except Exception as e:
        f_out.write(f" ❌ Error al leer CSV: {e}\\n")

def run_audit():
    if not DATA_DIR.exists():
        print(f"Error: La ruta {DATA_DIR} no existe.")
        sys.exit(1)
        
    h5_files = sorted(DATA_DIR.glob('*.h5'))
    csv_files = sorted(DATA_DIR.glob('*.csv'))
    
    print(f"Iniciando auditoría en: {DATA_DIR}")
    print(f"Archivos encontrados: {len(h5_files)} HDF5, {len(csv_files)} CSV.")
    print(f"Generando reporte, por favor espera... (los CSV grandes pueden tardar unos segundos)")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        f_out.write("="*80 + "\\n")
        f_out.write("REPORTE DE AUDITORÍA DEL DATASET DE INHIBIDORES\\n")
        f_out.write("="*80 + "\\n\\n")
        f_out.write(f"Directorio escaneado: {DATA_DIR}\\n")
        f_out.write(f"Total HDF5 : {len(h5_files)}\\n")
        f_out.write(f"Total CSV  : {len(csv_files)}\\n")
        
        for h5_file in h5_files:
            audit_h5_file(h5_file, f_out)
            
        for csv_file in csv_files:
            audit_csv_file(csv_file, f_out)
            
    print(f"\\n✅ Auditoría finalizada. Revisa el archivo: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_audit()