import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import sys
import torch
from pathlib import Path

# Add the parent directory to the path to import the detector client
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

try:
    # Try to import the YOLOv8 modules
    from detector_socket.detector_client import DetectionClient
except ImportError:
    print("Error importing DetectionClient. Make sure the detector_socket module is available.")
    sys.exit(1)

def benchmark_yolo_models(image_path, output_dir="benchmark_results_yolov8"):
    """
    Benchmark YOLOv8 models using the detector_server implementation.

    Args:
        image_path (str): Path to the test image
        output_dir (str): Directory to save results

    Returns:
        dict: Dictionary containing benchmark results
    """
    print(f"Benchmarking YOLOv8 models on image: {image_path}")

    # Verify that the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Convert BGR to RGB (YOLOv8 expects RGB format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get image dimensions
    height, width = img.shape[:2]

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define models to test
    models = [
        {
            'name': 'YOLOv8n',  # Nano - fastest but less accurate
            'model': 'yolov8n.pt',
            'color': '#3498db',  # Blue
            'input_size': 640
        },
        {
            'name': 'YOLOv8s',  # Small - balanced speed and accuracy
            'model': 'yolov8s.pt',
            'color': '#e74c3c',  # Red
            'input_size': 640
        },
        {
            'name': 'YOLOv8m',  # Medium - more accurate but slower
            'model': 'yolov8m.pt',
            'color': '#2ecc71',  # Green
            'input_size': 640
        }
    ]

    # Check if CUDA is available
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        device = "0"  # Use first GPU
    else:
        print("CUDA is not available. Using CPU")
        device = "cpu"

    # Results storage
    results = {}
    result_images = {}

    # Connect to detector server (for reference)
    print("Checking connection to detector server...")
    try:
        detector_client = DetectionClient(host="localhost", port=5555)
        print("Successfully connected to detector server")
        # Get labels from detector client
        labels = detector_client.labels
        detector_client.close()
    except Exception as e:
        print(f"Error connecting to detector server: {e}")
        print("Using default COCO labels")
        # Default COCO labels if detector server not available
        labels = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    # Import Ultralytics YOLO for direct model testing
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not found. Please install it with:")
        print("pip install ultralytics")
        return

    # Test each model
    for model_config in models:
        print(f"\nTesting model {model_config['name']}...")

        # Load model directly using ultralytics
        try:
            model = YOLO(model_config['model'])
        except Exception as e:
            print(f"Error loading model {model_config['name']}: {e}")
            continue

        # Set device
        model_name = model_config['name']

        # Make a copy of the original image
        img_copy = img_rgb.copy()

        # Warm-up run (not timed)
        print(f"Running warmup for {model_name}...")
        _ = model(img_copy, device=device)

        # Multiple timing runs for more stable measurements
        num_runs = 5
        print(f"Performing {num_runs} inference measurements...")
        inference_times = []

        for i in range(num_runs):
            start_time = time.time()
            results_model = model(img_copy, conf=0.25, device=device)
            end_time = time.time()
            current_time = end_time - start_time
            inference_times.append(current_time)
            print(f"  Run {i+1}/{num_runs}: {current_time:.4f}s")

        # Calculate statistics
        inference_time = sum(inference_times) / len(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        std_dev = np.std(inference_times)

        print(f"Inference time statistics:")
        print(f"  Average: {inference_time:.4f}s")
        print(f"  Minimum: {min_time:.4f}s")
        print(f"  Maximum: {max_time:.4f}s")
        print(f"  Standard deviation: {std_dev:.4f}s")

        # Extract the results from the last run
        if len(results_model) > 0:
            # Get all detections
            boxes = []
            confidences = []
            class_ids = []

            # Extract boxes, confidences, and class IDs
            for r in results_model:
                for box in r.boxes:
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Convert to x, y, w, h format
                    x, y = int(x1), int(y1)
                    w, h = int(x2 - x1), int(y2 - y1)

                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    boxes.append([x, y, w, h])
                    confidences.append(conf)
                    class_ids.append(cls_id)

            # Create visualization image (convert back to BGR for OpenCV)
            result_img = img.copy()

            # Generate colors for visualization
            np.random.seed(42)  # For consistent colors
            colors = np.random.randint(0, 255, size=(len(labels), 3), dtype=np.uint8)

            # Draw bounding boxes
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                x, y, w, h = box

                # Use class index as color index, with bounds checking
                color_idx = min(cls_id, len(colors)-1)
                color = colors[color_idx].tolist()

                # Get class label with bounds checking
                label = labels[cls_id] if cls_id < len(labels) else f"Class {cls_id}"

                # Draw rectangle and text
                cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(result_img, f"{label}: {conf:.2f}", (x, y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add performance info to the image
            fps = 1.0 / inference_time
            backend = "GPU (CUDA)" if has_cuda else "CPU"
            cv2.putText(result_img, f"{model_name} - {backend}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result_img, f"FPS: {fps:.2f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(result_img, f"Detections: {len(boxes)}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Save results
            results[model_name] = {
                'inference_time': inference_time,
                'fps': fps,
                'detections': len(boxes),
                'backend': backend
            }
            result_images[model_name] = result_img

            # Print results
            print(f"Inference time: {inference_time:.4f} seconds")
            print(f"FPS: {fps:.2f}")
            print(f"Objects detected: {len(boxes)}")

            # Save individual result image
            cv2.imwrite(os.path.join(output_dir, f"{model_name}_result.png"), result_img)
        else:
            print(f"No detections for {model_name}")

    # Create comparison image
    if result_images:
        img_height, img_width = img.shape[:2]
        max_height = img_height
        total_width = img_width * len(result_images)

        # Create combined image
        comparison_img = np.zeros((max_height, total_width, 3), dtype=np.uint8)

        # Combine images side by side
        x_offset = 0
        for model_name, img in result_images.items():
            comparison_img[0:img.shape[0], x_offset:x_offset+img.shape[1], :] = img
            x_offset += img.shape[1]

        # Save comparison image
        cv2.imwrite(os.path.join(output_dir, "yolov8_comparison.png"), comparison_img)

    # Generate performance graphs
    generate_performance_graphs(results, output_dir)

    return results

def generate_performance_graphs(resultados, diretorio_saida):
    """
    Gera gráficos comparativos de desempenho dos modelos YOLOv8.

    Args:
        resultados (dict): Dicionário com resultados do benchmark
        diretorio_saida (str): Diretório para salvar os gráficos
    """
    # Extrair dados
    modelos = list(resultados.keys())
    tempos_inferencia = [resultados[modelo]['inference_time'] for modelo in modelos]
    valores_fps = [resultados[modelo]['fps'] for modelo in modelos]
    contagens_deteccoes = [resultados[modelo]['detections'] for modelo in modelos]
    backends = [resultados[modelo]['backend'] for modelo in modelos]

    # Cores para cada modelo
    modelo_cores = {
        'YOLOv8n': '#3498db',  # Azul
        'YOLOv8s': '#e74c3c',  # Vermelho
        'YOLOv8m': '#2ecc71',  # Verde
    }
    cores = [modelo_cores.get(modelo, '#95a5a6') for modelo in modelos]

    # Precisão estimada (mAP@0.5-0.95 no COCO val2017)
    # Fonte: https://docs.ultralytics.com/models/yolov8/
    precisao = {
        'YOLOv8n': 37.3,
        'YOLOv8s': 44.9,
        'YOLOv8m': 50.2,
        'YOLOv8l': 52.9,
        'YOLOv8x': 53.9
    }

    # Figura 1: Tempo de Processamento vs FPS
    plt.figure(figsize=(14, 10))

    # Gráfico de barras para tempo de inferência
    plt.subplot(2, 1, 1)
    barras = plt.bar(modelos, tempos_inferencia, color=cores, alpha=0.8)
    plt.title('Tempo de Processamento por Frame', fontsize=14, fontweight='bold')
    plt.ylabel('Tempo (segundos)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar valores nas barras
    # Adicionar valores nas barras
    for barra, tempo_val in zip(barras, tempos_inferencia):
        plt.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.001,  # Reduzir de 0.01 para 0.001
                f'{tempo_val:.4f}s', ha='center', fontweight='bold')

    # Adicionar informação sobre backend (CPU/GPU)
    for i, backend in enumerate(backends):
        plt.text(i, 0.01, backend, ha='center', color='white',
                fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))

    # Gráfico de barras para FPS
    plt.subplot(2, 1, 2)
    barras = plt.bar(modelos, valores_fps, color=cores, alpha=0.8)
    plt.title('Taxa de Frames por Segundo (FPS)', fontsize=14, fontweight='bold')
    plt.ylabel('FPS', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar valores nas barras
    for barra, fps in zip(barras, valores_fps):
        plt.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.2,
                f'{fps:.2f} FPS', ha='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Comparação de Desempenho entre Modelos YOLOv8', fontsize=16, fontweight='bold')

    # Salvar figura 1
    plt.savefig(os.path.join(diretorio_saida, 'yolov8_desempenho_tempo_fps.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(diretorio_saida, 'yolov8_desempenho_tempo_fps.png'), format='png', dpi=300, bbox_inches='tight')

    # Figura 2: Precisão vs Velocidade vs Detecções
    plt.figure(figsize=(14, 10))

    # Gráfico de dispersão para Precisão vs FPS
    plt.subplot(2, 1, 1)
    for modelo in modelos:
        plt.scatter(precisao.get(modelo, 0), resultados[modelo]['fps'],
                  s=100, color=modelo_cores.get(modelo, '#95a5a6'), label=modelo, alpha=0.8)
        plt.annotate(modelo,
                   (precisao.get(modelo, 0), resultados[modelo]['fps']),
                   xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

    plt.title('Relação entre Precisão e Velocidade', fontsize=14, fontweight='bold')
    plt.xlabel('Precisão (mAP@0.5-0.95)', fontsize=12)
    plt.ylabel('Velocidade (FPS)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Gráfico de barras para Número de Detecções
    plt.subplot(2, 1, 2)
    barras = plt.bar(modelos, contagens_deteccoes, color=cores, alpha=0.8)
    plt.title('Número de Objetos Detectados', fontsize=14, fontweight='bold')
    plt.ylabel('Quantidade', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adicionar valores nas barras
    for barra, contagem in zip(barras, contagens_deteccoes):
        plt.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.1,
                f'{contagem}', ha='center', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('Análise de Eficácia dos Modelos YOLOv8', fontsize=16, fontweight='bold')

    # Salvar figura 2
    plt.savefig(os.path.join(diretorio_saida, 'yolov8_desempenho_precisao.pdf'), format='pdf', bbox_inches='tight')
    plt.savefig(os.path.join(diretorio_saida, 'yolov8_desempenho_precisao.png'), format='png', dpi=300, bbox_inches='tight')

    # Gerar mapa de calor para tempos de inferência
    plt.figure(figsize=(10, 6))
    dados = np.array([tempos_inferencia])
    plt.imshow(dados, cmap='viridis')
    plt.colorbar(label='Tempo de Inferência (segundos)')
    plt.xticks(range(len(modelos)), modelos, rotation=45)
    plt.yticks([])
    plt.title('Comparação de Tempo de Inferência')
    plt.tight_layout()
    plt.savefig(os.path.join(diretorio_saida, 'mapa_calor_tempos.png'), dpi=300)

    # Criar tabela comparativa em CSV
    with open(os.path.join(diretorio_saida, 'comparacao_modelos.csv'), 'w') as f:
        f.write("Modelo,Tempo de Inferência (s),FPS,Detecções,Backend\n")
        for modelo in modelos:
            r = resultados[modelo]
            f.write(f"{modelo},{r['inference_time']:.4f},{r['fps']:.2f},{r['detections']},{r['backend']}\n")

    # Imprimir resumo
    print("\nResumo do Benchmark:")
    print(f"{'Modelo':<12} {'Tempo (s)':<10} {'FPS':<8} {'Detecções':<10} {'Backend':<6}")
    print("-" * 50)
    for modelo in modelos:
        r = resultados[modelo]
        print(f"{modelo:<12} {r['inference_time']:<10.4f} {r['fps']:<8.2f} {r['detections']:<10} {r['backend']:<6}")

    print(f"\nResultados detalhados e gráficos salvos em {diretorio_saida}/")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 models on a test image")
    parser.add_argument("-i", "--image", default="../_out/episode_3360/frame_camera/000970.png",
                      help="Path to test image")
    parser.add_argument("-o", "--output", default="benchmark_results_yolov8",
                      help="Directory to save results")

    args = parser.parse_args()

    benchmark_yolo_models(args.image, args.output)
