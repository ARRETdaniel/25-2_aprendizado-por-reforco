import cv2
import numpy as np
import time
import os
import matplotlib.pyplot as plt

def benchmark_yolo_models(image_path):
    """
    Compara o desempenho de diferentes modelos YOLO usando uma imagem real
    e gera gráficos com os resultados.
    """
    print(f"Benchmarking YOLO models on image: {image_path}")

    # Verificar se a imagem existe
    if not os.path.exists(image_path):
        print(f"Erro: Imagem não encontrada em {image_path}")
        return

    # Carregar a imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não foi possível ler a imagem em {image_path}")
        return

    # Obter dimensões da imagem
    height, width = img.shape[:2]

    # Carregar as labels
    labels_path = '../yolov3-coco/coco-labels'
    if not os.path.exists(labels_path):
        print(f"Erro: Arquivo de labels não encontrado em {labels_path}")
        return

    labels = open(labels_path).read().strip().split('\n')
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Definir modelos a serem testados
    models = [
        {
            'name': 'YOLOv3',
            'cfg': '../yolov3-coco/yolov3.cfg',
            'weights': '../yolov3-coco/yolov3.weights',
            'color': '#3498db',  # Azul
            'input_size': (416, 416)
        },
        {
            'name': 'YOLOv4',
            'cfg': '../yolov4-coco/yolov4.cfg',
            'weights': '../yolov4-coco/yolov4.weights',
            'color': '#e74c3c',  # Vermelho
            'input_size': (416, 416)
        },
        {
            'name': 'YOLOv3-tiny',
            'cfg': '../yolov3-coco/yolov3-tiny.cfg',
            'weights': '../yolov3-coco/yolov3-tiny.weights',
            'color': '#2ecc71',  # Verde
            'input_size': (416, 416)
        }
    ]

    # Resultados
    results = {}
    result_images = {}

    # Executar testes para cada modelo
    for model in models:
        print(f"\nTestando modelo {model['name']}...")

        # Verificar se os arquivos do modelo existem
        if not os.path.exists(model['cfg']) or not os.path.exists(model['weights']):
            print(f"Erro: Arquivos do modelo {model['name']} não encontrados")
            continue

        # Carregar o modelo
        net = cv2.dnn.readNetFromDarknet(model['cfg'], model['weights'])

        # Tentar usar GPU aceleração
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            # Check if CUDA was actually set by trying to get the backend info
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                backend = "CUDA"
            else:
                backend = "CPU"
                print(f"CUDA não disponível. Usando CPU para {model['name']}")
        except Exception as e:
            backend = "CPU"
            print(f"Usando CPU para {model['name']}: {str(e)}")

        # Obter camadas de saída
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

        # Executar detecção e medir tempo
        img_copy = img.copy()
        blob = cv2.dnn.blobFromImage(img_copy, 1/255.0, model['input_size'],
                                   swapRB=True, crop=False)
        net.setInput(blob)

        # Warmup run to initialize any on-demand operations
        print(f"Executando warmup para {model['name']}...")
        _ = net.forward(output_layers)  # Warmup run, not timed

        # Multiple timing runs for more stable measurements
        num_runs = 5
        print(f"Realizando {num_runs} medições de inferência...")
        inference_times = []

        for i in range(num_runs):
            start_time = time.time()
            outs = net.forward(output_layers)
            end_time = time.time()
            current_time = end_time - start_time
            inference_times.append(current_time)
            print(f"  Run {i+1}/{num_runs}: {current_time:.4f}s")

        # Calculate statistics
        inference_time = sum(inference_times) / len(inference_times)
        min_time = min(inference_times)
        max_time = max(inference_times)
        std_dev = np.std(inference_times)

        print(f"Estatísticas de tempo de inferência:")
        print(f"  Média: {inference_time:.4f}s")
        print(f"  Mínimo: {min_time:.4f}s")
        print(f"  Máximo: {max_time:.4f}s")
        print(f"  Desvio padrão: {std_dev:.4f}s")

        # Processar detecções
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Limiar de confiança
                    # Coordenadas da caixa (formato YOLO: centro_x, centro_y, largura, altura)
                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, box_width, box_height) = box.astype("int")

                    # Converter para coordenadas de canto (x, y, largura, altura)
                    x = int(center_x - (box_width / 2))
                    y = int(center_y - (box_height / 2))

                    boxes.append([x, y, int(box_width), int(box_height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Aplicar Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Desenhar caixas na imagem
        if len(indices) > 0:
            for i in indices.flatten():
                # Get the bounding box coordinates
                (x, y, w, h) = boxes[i]

                # Make sure class_id is within range of labels
                class_id = class_ids[i]
                if class_id < len(labels):
                    label_text = f"{labels[class_id]}: {confidences[i]:.2f}"
                else:
                    label_text = f"Class {class_id}: {confidences[i]:.2f}"

                # Get color (ensure it's within range)
                color_index = class_id % len(colors)
                color = [int(c) for c in colors[color_index]]

                # Draw rectangle and label
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img_copy, label_text, (x, y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Adicionar informações de desempenho à imagem
        fps = 1.0 / inference_time
        cv2.putText(img_copy, f"{model['name']} - {backend}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img_copy, f"FPS: {fps:.2f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img_copy, f"Deteccoes: {len(indices) if len(indices) > 0 else 0}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Guardar resultados
        results[model['name']] = {
            'inference_time': inference_time,
            'fps': fps,
            'detections': len(indices) if len(indices) > 0 else 0,
            'backend': backend
        }
        result_images[model['name']] = img_copy

        # Exibir resultados no console
        print(f"Tempo de inferência: {inference_time:.4f} segundos")
        print(f"FPS: {fps:.2f}")
        print(f"Objetos detectados: {len(indices) if len(indices) > 0 else 0}")

    # Salvar as imagens resultado
    output_dir = "benchmark_results_new-drawing"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Criar comparação visual
    img_height, img_width = img.shape[:2]
    max_height = img_height
    total_width = img_width * len(result_images)

    # Criar imagem combinada
    comparison_img = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    # Combinar imagens lado a lado
    x_offset = 0
    for model_name, img in result_images.items():
        comparison_img[0:img.shape[0], x_offset:x_offset+img.shape[1], :] = img
        x_offset += img.shape[1]

    # Salvar a comparação visual
    cv2.imwrite(os.path.join(output_dir, "yolo_comparison.png"), comparison_img)

    # Gerar gráficos
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
    for barra, tempo_val in zip(barras, tempos_inferencia):
        plt.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.01,
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

    parser = argparse.ArgumentParser(description="Benchmark de modelos YOLO com imagem real")
    parser.add_argument("-i", "--image", default="../_out/episode_3360/frame_camera/000970.png",
                      help="Caminho para imagem de teste")
    args = parser.parse_args()

    benchmark_yolo_models(args.image)
