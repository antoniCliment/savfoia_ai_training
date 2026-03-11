import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_training_data(data_path='training_data.csv', output_path='training_data_visualization.png'):
    """
    Crea una visualización completa de los datos de entrenamiento para entender
    las relaciones entre las bandas espectrales, el rendimiento y la dosis de nitrógeno.
    """
    if not os.path.exists(data_path):
        print(f"Error: El archivo {data_path} no existe.")
        return

    # Cargar datos
    df = pd.read_csv(data_path)
    
    # Calcular NDVI si no existe 
    if 'ndvi' not in df.columns:
        df['ndvi'] = (df['B08'] - df['B04']) / (df['B08'] + df['B04'] + 1e-8)

    # Configurar estilo
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Análisis de Datos de Entrenamiento: Dosis de Nitrógeno', fontsize=20, fontweight='bold', y=0.98)

    # 1. Distribución de la Dosis de Nitrógeno
    axes[0, 0].hist(df['nitrogen_dosis'], bins=12, color='#3498db', edgecolor='white', alpha=0.8)
    axes[0, 0].set_title('Distribución de Dosis (kg/ha)', fontsize=13)
    axes[0, 0].set_xlabel('Nitrógeno (kg/ha)')
    axes[0, 0].set_ylabel('Frecuencia')

    # 2. Relación NDVI vs Dosis de Nitrógeno
    scatter1 = axes[0, 1].scatter(df['ndvi'], df['nitrogen_dosis'], c=df['rendimiento_objetivo'], 
                                   cmap='viridis', s=60, alpha=0.7, edgecolor='k')
    axes[0, 1].set_title('NDVI vs Dosis', fontsize=13)
    axes[0, 1].set_xlabel('NDVI')
    axes[0, 1].set_ylabel('Nitrógeno (kg/ha)')
    cbar1 = fig.colorbar(scatter1, ax=axes[0, 1])
    cbar1.set_label('Rendimiento (t/ha)')

    # 3. Relación Rendimiento Objetivo vs Dosis de Nitrógeno
    scatter2 = axes[0, 2].scatter(df['rendimiento_objetivo'], df['nitrogen_dosis'], c=df['ndvi'], 
                                   cmap='coolwarm', s=60, alpha=0.7, edgecolor='k')
    axes[0, 2].set_title('Rendimiento vs Dosis', fontsize=13)
    axes[0, 2].set_xlabel('Rendimiento (t/ha)')
    axes[0, 2].set_ylabel('Nitrógeno (kg/ha)')
    cbar2 = fig.colorbar(scatter2, ax=axes[0, 2])
    cbar2.set_label('NDVI')

    # 4. Matriz de Correlación (Manual con Matplotlib)
    cols_corr = ['B02', 'B03', 'B04', 'B08', 'rendimiento_objetivo', 'nitrogen_dosis', 'ndvi']
    corr = df[cols_corr].corr()
    im = axes[1, 0].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 0].set_title('Matriz de Correlación', fontsize=13)
    axes[1, 0].set_xticks(np.arange(len(cols_corr)))
    axes[1, 0].set_yticks(np.arange(len(cols_corr)))
    axes[1, 0].set_xticklabels(cols_corr, rotation=45, ha='right')
    axes[1, 0].set_yticklabels(cols_corr)
    fig.colorbar(im, ax=axes[1, 0])
    
    # Añadir valores a la matriz
    for i in range(len(cols_corr)):
        for j in range(len(cols_corr)):
            text = axes[1, 0].text(j, i, f"{corr.iloc[i, j]:.2f}",
                                   ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.7 else "white",
                                   fontsize=9)

    # 5. Banda Roja (B04) vs Dosis (Relacionada con NDVI)
    axes[1, 1].scatter(df['B04'], df['nitrogen_dosis'], color='#e74c3c', s=60, alpha=0.7, edgecolor='k')
    axes[1, 1].set_title('Banda Roja (B04) vs Dosis', fontsize=13)
    axes[1, 1].set_xlabel('Reflectancia B04')
    axes[1, 1].set_ylabel('Nitrógeno (kg/ha)')

    # 6. Banda NIR (B08) vs Dosis
    axes[1, 2].scatter(df['B08'], df['nitrogen_dosis'], color='#9b59b6', s=60, alpha=0.7, edgecolor='k')
    axes[1, 2].set_title('Banda NIR (B08) vs Dosis', fontsize=13)
    axes[1, 2].set_xlabel('Reflectancia B08')
    axes[1, 2].set_ylabel('Nitrógeno (kg/ha)')

    # Ajustar diseño
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Guardar la imagen
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"Visualización guardada con éxito en: {output_path}")
    plt.close()

if __name__ == "__main__":
    visualize_training_data()
