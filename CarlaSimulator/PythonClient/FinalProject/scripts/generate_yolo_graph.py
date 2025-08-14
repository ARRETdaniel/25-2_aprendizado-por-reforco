import matplotlib.pyplot as plt
import numpy as np
import os

def generate_performance_graph():
    """
    Gera um gráfico comparativo de desempenho entre YOLOv3 e YOLOv4
    """
    # Dados de desempenho
    models = ['YOLOv3', 'YOLOv4', 'YOLOv3-tiny']
    
    # Tempos de processamento em segundos
    processing_times = [0.4807, 0.5104, 0.15]  # YOLOv3-tiny é aproximadamente 3-4x mais rápido
    
    # Convertendo para FPS (Frames Por Segundo)
    fps = [1/time for time in processing_times]
    
    # Precisão estimada (baseado em benchmarks gerais)
    accuracy = [73, 75, 58]  # Valores aproximados para COCO mAP
    
    # Cores para cada modelo
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # Criando a figura com dois subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico de barras para tempo de processamento
    ax1.bar(models, processing_times, color=colors, alpha=0.7)
    ax1.set_title('Tempo de Processamento por Frame', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Tempo (segundos)', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar os valores nas barras
    for i, v in enumerate(processing_times):
        ax1.text(i, v + 0.02, f'{v:.3f}s', ha='center', fontweight='bold')
    
    # Gráfico de barras para FPS
    ax2.bar(models, fps, color=colors, alpha=0.7)
    ax2.set_title('Taxa de Frames por Segundo (FPS)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('FPS', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar os valores nas barras
    for i, v in enumerate(fps):
        ax2.text(i, v + 0.3, f'{v:.1f} FPS', ha='center', fontweight='bold')
    
    # Adicionar texto explicativo sobre precisão
    plt.figtext(0.5, 0.01, 
                f'Precisão (mAP no COCO): YOLOv3: {accuracy[0]}%, YOLOv4: {accuracy[1]}%, YOLOv3-tiny: {accuracy[2]}%', 
                ha='center', fontsize=10, bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.suptitle('Comparação de Desempenho entre Versões YOLO em CPU', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Salvar a figura
    output_path = 'yolo_performance_comparison.pdf'
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    
    # Salvar também como PNG para visualização rápida
    plt.savefig('yolo_performance_comparison.png', format='png', bbox_inches='tight', dpi=300)
    
    print(f"Gráfico salvo como {output_path}")
    plt.show()

if __name__ == "__main__":
    generate_performance_graph()