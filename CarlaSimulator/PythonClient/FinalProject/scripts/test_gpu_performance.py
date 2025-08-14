import cv2
import numpy as np
import time
import os
import sys

def test_gpu_compatibility():
    """Verifica a compatibilidade do sistema com aceleração GPU."""
    print("=== Teste de Compatibilidade GPU ===")
    print("OpenCV versão:", cv2.__version__)

    # Verificar suporte a CUDA
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"Dispositivos CUDA disponíveis: {cuda_devices}")

    if cuda_devices > 0:
        print("Status: ✅ GPU NVIDIA detectada e compatível com CUDA")
        for i in range(cuda_devices):
            device_info = cv2.cuda.getDevice()
            print(f"   GPU #{i}: {cv2.cuda.printShortCudaDeviceInfo(i)}")
    else:
        print("Status: ❌ Nenhuma GPU NVIDIA com CUDA detectada")
        print("Verificando outros backends...")

        build_info = cv2.getBuildInformation()

        # Verificar suporte OpenCL (AMD/Intel GPUs)
        if "OpenCL:" in build_info and "YES" in build_info.split("OpenCL:")[1].split("\n")[0]:
            print("Status: ✅ OpenCL disponível - GPUs AMD/Intel podem ser usadas")

            # Verificar dispositivos OpenCL disponíveis
            try:
                cv2.ocl.setUseOpenCL(True)
                if cv2.ocl.haveOpenCL():
                    print(f"   OpenCL disponível: {cv2.ocl.haveOpenCL()}")
                    print(f"   OpenCL em uso: {cv2.ocl.useOpenCL()}")
                    device_info = cv2.ocl.Device.getDefault().name()
                    print(f"   Dispositivo OpenCL: {device_info}")
                else:
                    print("   OpenCL não disponível no sistema")
            except:
                print("   Não foi possível obter informações sobre dispositivos OpenCL")
        else:
            print("Status: ❌ OpenCL não disponível")
            print("Status final: ⚠️ Sistema compatível apenas com CPU")

def benchmark_yolo_performance(use_gpu=True):
    """Testa a performance do YOLO com e sem GPU."""
    print("\n=== Benchmark de Performance YOLO ===")

    # Carregar modelo
    model_config = '../yolov3-coco/yolov3.cfg'
    model_weights = '../yolov3-coco/yolov3.weights'

    try:
        # Criar uma imagem de teste simples
        img = np.zeros((600, 800, 3), dtype=np.uint8)

        # Carregar a rede
        net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

        if use_gpu:
            # Tentar configurar para GPU
            try:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                device = "GPU (CUDA)"
            except:
                try:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                    device = "GPU (OpenCL)"
                except:
                    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    device = "CPU (GPU não disponível)"
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            device = "CPU"

        # Obter as camadas de saída
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

        # Preparar a entrada
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        # Medir o tempo
        start_time = time.time()
        num_iterations = 10  # Executar várias vezes para medição mais precisa

        for _ in range(num_iterations):
            _ = net.forward(output_layers)

        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        fps = 1.0 / avg_time

        print(f"Dispositivo: {device}")
        print(f"Tempo médio de inferência: {avg_time:.4f} segundos")
        print(f"FPS estimado: {fps:.2f}")
        print(f"Status para uso em tempo real: {'✅ APROVADO' if fps > 10 else '❌ REPROVADO'} (limite: 10 FPS)")

    except Exception as e:
        print(f"Erro durante o benchmark: {e}")

def benchmark_yolo_tiny():
    """Testa a performance do YOLOv3-tiny com diferentes configurações."""
    print("\n=== Benchmark YOLOv3-tiny ===")

    # Baixar arquivos se necessário
    model_config = '../yolov3-coco/yolov3-tiny.cfg'
    model_weights = '../yolov3-coco/yolov3-tiny.weights'

    # Verificar e baixar arquivos
    import os
    import urllib.request

    if not os.path.exists('../yolov3-coco'):
        os.makedirs('./yolov3-coco')

    if not os.path.exists(model_config):
        print("Baixando yolov3-tiny.cfg...")
        urllib.request.urlretrieve("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg", model_config)

    if not os.path.exists(model_weights):
        print("Baixando yolov3-tiny.weights...")
        urllib.request.urlretrieve("https://pjreddie.com/media/files/yolov3-tiny.weights", model_weights)

    try:
        # Criar imagem de teste
        img = np.zeros((600, 800, 3), dtype=np.uint8)

        # Testar com diferentes configurações
        configs = [
            {"nome": "YOLOv3-tiny CPU", "backend": cv2.dnn.DNN_BACKEND_DEFAULT, "target": cv2.dnn.DNN_TARGET_CPU, "size": (416, 416)},
            {"nome": "YOLOv3-tiny OpenCL", "backend": cv2.dnn.DNN_BACKEND_OPENCV, "target": cv2.dnn.DNN_TARGET_OPENCL, "size": (416, 416)},
            {"nome": "YOLOv3-tiny OpenCL (320x320)", "backend": cv2.dnn.DNN_BACKEND_OPENCV, "target": cv2.dnn.DNN_TARGET_OPENCL, "size": (320, 320)},
            {"nome": "YOLOv3-tiny OpenCL (224x224)", "backend": cv2.dnn.DNN_BACKEND_OPENCV, "target": cv2.dnn.DNN_TARGET_OPENCL, "size": (224, 224)}
        ]

        # Se CUDA disponível, adicionar configurações
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            configs.extend([
                {"nome": "YOLOv3-tiny CUDA", "backend": cv2.dnn.DNN_BACKEND_CUDA, "target": cv2.dnn.DNN_TARGET_CUDA, "size": (416, 416)},
                {"nome": "YOLOv3-tiny CUDA (320x320)", "backend": cv2.dnn.DNN_BACKEND_CUDA, "target": cv2.dnn.DNN_TARGET_CUDA, "size": (320, 320)},
                {"nome": "YOLOv3-tiny CUDA (224x224)", "backend": cv2.dnn.DNN_BACKEND_CUDA, "target": cv2.dnn.DNN_TARGET_CUDA, "size": (224, 224)}
            ])

        for config in configs:
            print(f"\nTestando {config['nome']} com resolução {config['size']}...")

            # Carregar rede
            net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

            try:
                net.setPreferableBackend(config['backend'])
                net.setPreferableTarget(config['target'])
            except Exception as e:
                print(f"Erro ao configurar backend {config['nome']}: {e}")
                continue

            # Obter camadas de saída
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

            # Preparar entrada
            blob = cv2.dnn.blobFromImage(img, 1/255.0, config['size'], swapRB=True, crop=False)
            net.setInput(blob)

            # Medir tempo
            start_time = time.time()
            num_iterations = 20

            for _ in range(num_iterations):
                _ = net.forward(output_layers)

            end_time = time.time()
            avg_time = (end_time - start_time) / num_iterations
            fps = 1.0 / avg_time

            # Simular frame skipping (2x FPS)
            fps_with_skipping = fps * 2

            print(f"Tempo médio: {avg_time:.4f} segundos")
            print(f"FPS estimado: {fps:.2f}")
            print(f"FPS com frame skipping: {fps_with_skipping:.2f}")
            print(f"Status: {'✅ APROVADO' if fps_with_skipping > 10 else '❌ REPROVADO'} (limite: 10 FPS)")

    except Exception as e:
        print(f"Erro durante o benchmark do YOLOv3-tiny: {e}")

def check_build_info():
    print("\n=== Informações de Compilação do OpenCV ===")
    build_info = cv2.getBuildInformation()

    # Verificar suporte a CUDA
    cuda_info = "CUDA:" in build_info and "YES" in build_info.split("CUDA:")[1].split("\n")[0]
    print(f"CUDA no build: {'✅ SIM' if cuda_info else '❌ NÃO'}")

    # Verificar versão do CUDA
    if cuda_info:
        try:
            cuda_version = build_info.split("CUDA Version:")[1].split("\n")[0].strip()
            print(f"Versão CUDA: {cuda_version}")
        except:
            print("Não foi possível determinar a versão CUDA")

    # Verificar suporte ao módulo DNN com CUDA
    dnn_cuda = "DNN_CUDA" in build_info and "YES" in build_info.split("DNN_CUDA:")[1].split("\n")[0]
    print(f"DNN com suporte CUDA: {'✅ SIM' if dnn_cuda else '❌ NÃO'}")

    # Exibir as primeiras linhas com informações relevantes
    print("\nTrechos relevantes do build:")
    for line in build_info.split('\n'):
        if any(x in line for x in ["CUDA", "GPU", "OpenCL", "DNN", "NVIDIA"]):
            print(f"  {line.strip()}")

def check_cuda_environment():
    print("\n=== Verificação do Ambiente CUDA ===")

    # Verificar variáveis de ambiente
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    print(f"CUDA_HOME: {cuda_home or 'Não definido'}")

    # Verificar se o CUDA está no PATH
    path = os.environ.get('PATH', '')
    cuda_in_path = any('cuda' in p.lower() for p in path.split(os.pathsep))
    print(f"CUDA no PATH: {'✅ SIM' if cuda_in_path else '❌ NÃO'}")

    # Verificar se arquivos DLL do CUDA estão acessíveis
    cuda_dll_path = os.path.join(cuda_home, 'bin') if cuda_home else None
    if cuda_dll_path and os.path.exists(cuda_dll_path):
        dlls = [f for f in os.listdir(cuda_dll_path) if f.lower().endswith('.dll') and 'cuda' in f.lower()]
        print(f"DLLs CUDA encontradas: {len(dlls)}")
    else:
        print("Pasta de DLLs CUDA não encontrada")

    # Verificar módulo Python CUDA
    try:
        import torch
        print(f"PyTorch CUDA disponível: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch não instalado (usado para teste secundário de CUDA)")

if __name__ == "__main__":
    test_gpu_compatibility()
    check_build_info()
    check_cuda_environment()

    print("\nTestando performance com modelos padrão...")
    benchmark_yolo_performance(use_gpu=False)
    benchmark_yolo_performance(use_gpu=True)

    print("\nTestando performance com YOLOv3-tiny...")
    benchmark_yolo_tiny()
