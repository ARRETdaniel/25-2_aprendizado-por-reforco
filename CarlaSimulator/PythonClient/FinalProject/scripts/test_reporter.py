import sys
sys.path.append(r'C:\Users\danie\Documents\Documents\CURSOS\Self-Driving_Cars_Specialization\CarlaSimulator\PythonClient\FinalProject')
from results_reporter import ResultsReporter
import os
import shutil
import matplotlib.pyplot as plt
import time

def test_with_real_data():
    """
    Testa o gerador de relatórios usando dados reais coletados anteriormente
    na pasta metrics_output. Cria um relatório em uma nova pasta de saída.
    """
    # Diretório com os dados reais
    real_metrics_dir = "metrics_output"

    # Diretório para o novo relatório
    report_dir = "real_data_report"

    # Verificar se temos os dados necessários
    if not os.path.exists(real_metrics_dir):
        print(f"ERRO: Diretório {real_metrics_dir} não encontrado. Execute a simulação primeiro para coletar dados.")
        return

    required_files = [
        'summary_stats.csv',
        'detection_metrics.csv',
        'warning_metrics.csv',
        'class_accuracy.csv',
        'distance_metrics.csv'
    ]

    missing_files = [f for f in required_files if not os.path.exists(os.path.join(real_metrics_dir, f))]
    if missing_files:
        print(f"AVISO: Alguns arquivos necessários não foram encontrados: {missing_files}")
        print("O relatório pode não estar completo.")

    # Criar diretório do relatório com tratamento de erros
    if os.path.exists(report_dir):
        try:
            # Primeiro tenta excluir normalmente
            shutil.rmtree(report_dir)
        except PermissionError:
            print(f"AVISO: Não foi possível remover o diretório {report_dir} (acesso negado).")
            print("Tentando método alternativo...")

            # Tenta renomear o diretório e depois excluir
            try:
                new_name = f"{report_dir}_old_{time.time()}"
                os.rename(report_dir, new_name)
                print(f"Diretório renomeado para {new_name}")
            except:
                # Se não conseguir renomear, tenta criar um diretório com nome alternativo
                report_dir = f"real_data_report_{int(time.time())}"
                print(f"Usando diretório alternativo: {report_dir}")

    # Tenta criar o diretório se ainda não existir
    try:
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
    except Exception as e:
        print(f"ERRO ao criar diretório {report_dir}: {e}")
        # Usa um diretório temporário como fallback
        import tempfile
        report_dir = tempfile.mkdtemp(prefix="report_")
        print(f"Usando diretório temporário: {report_dir}")

    # Importar a classe atualizada PerformanceMetrics para recriar os gráficos traduzidos
    try:
        from performance_metrics import PerformanceMetrics
        print("Recriando gráficos com legendas em português...")

        # Instanciamos apenas para acessar a função de visualização
        # (os dados reais já foram coletados)
        metrics = PerformanceMetrics(output_dir=real_metrics_dir)

        # Queremos forçar a renderização dos gráficos com as alterações feitas
        metrics.visualize_metrics()
        print("Gráficos recriados com sucesso!")

    except Exception as e:
        print(f"AVISO: Não foi possível recriar os gráficos: {e}")
        print("Usando os gráficos existentes...")

    # Criar relatório usando os dados reais
    try:
        reporter = ResultsReporter(metrics_dir=real_metrics_dir, report_dir=report_dir)

        # Gerar o relatório
        reporter.generate_report()

        print(f"\nRelatório gerado com sucesso em {report_dir}/relatorio_desempenho.html")
        print("Abra este arquivo em um navegador para visualizar o relatório completo.")
    except Exception as e:
        print(f"ERRO ao gerar relatório: {e}")

    print("\nVerifique se as seguintes melhorias foram aplicadas:")
    print("1. Gráficos com títulos e legendas em português")
    print("2. Nomes das classes de objetos em português (ex: 'Veículo' em vez de 'Class 2')")
    print("3. Rótulos das faixas de distância em português (Próximo, Médio, Distante)")
    print("4. Legendas corretamente traduzidas (ex: 'Precisão' em vez de 'Precision')")
    print("\nSe houver algum problema na visualização, revise as alterações feitas em performance_metrics.py.")

def verify_graphs_language():
    """
    Verifica se as legendas dos gráficos estão em português.
    """
    metrics_dir = "metrics_output"

    # Lista de verificação para arquivos de gráficos
    graphs = [
        'performance_metrics.png',
        'detection_times_histogram.png',
        'class_precision_metrics.png',
        'distance_metrics.png'
    ]

    print("\nVerificando arquivos de gráficos:")
    for graph in graphs:
        path = os.path.join(metrics_dir, graph)
        if os.path.exists(path):
            print(f"✓ {graph} encontrado")
            # Idealmente aqui poderíamos extrair e verificar o texto da imagem,
            # mas isso exigiria OCR e está além do escopo deste script
        else:
            print(f"✗ {graph} não encontrado")

    print("\nINSTRUÇÕES DE VERIFICAÇÃO MANUAL:")
    print("Por favor, verifique manualmente os gráficos gerados na pasta 'metrics_output'")
    print("e confirme se os títulos, eixos e legendas estão todos em português.")

if __name__ == "__main__":
    print("=== Teste com Dados Reais ===")
    test_with_real_data()
    verify_graphs_language()
