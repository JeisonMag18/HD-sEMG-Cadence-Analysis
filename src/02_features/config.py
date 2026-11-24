"""Parâmetros centrais. Edite DATASET caso a pasta esteja fora do PyCharm."""
from pathlib import Path
import os

# Configuração dos diretórios do conjunto de dados e dos resultados.
#
# Por padrão, o conjunto de dados deve estar em uma pasta chamada
# "Dataset", localizada no diretório do projeto
# Alternativamente, a variável de ambiente IA753_DATASET permite
# especificar outro caminho sem modificar o código.

PROJETO = Path(__file__).resolve().parent
_dataset_padrao = PROJETO / "Dataset"

if not _dataset_padrao.is_dir() and (PROJETO.parent / "Dataset").is_dir():
    _dataset_padrao = PROJETO.parent / "Dataset"
DATASET = Path(os.environ.get("IA753_DATASET", str(_dataset_padrao))).expanduser()
RESULTADOS = PROJETO / "resultados_ia753"
CACHE = RESULTADOS / "features_welch"
FIGURAS = RESULTADOS / "figuras"
for pasta in (RESULTADOS, CACHE, FIGURAS):
    pasta.mkdir(parents=True, exist_ok=True)

FS_EMG = 2052.52
FS_ANG = 100.0
INTERVALO_INICIO = 5.0  # Inicio de suporte temporal para o primeiro PSD
INTERVALO_FIM = 40.0    # Ultimo instante de referência
JANELA_RMS = 0.300    # 300 ms
JANELA_PSD = 2.000    # Bloco de 2 seg; Welch com segmentos Hann de ~1 seg e 50% overlap
PASSO = 0.250         # deslocamento entre estimativas da análise; sobreposição entre janelas
FAIXA_ESPECTRAL = (10.0, 500.0)
# Deslocamento não inferido; em princípio os sistemas iniciam sincronizados,
# mas verificar artefatos/registro e eventuais correções antes de conclusões.
DESLOCAMENTO_ANGULOS = 0.0

# Depois de confirmar, preencher com índices base-zero 0..63 em cada posição (8,8).
MAPA_EDC = None
MAPA_FDS = None

# A documentação descreve grupos XYZ de polegar, indicador, médio, anelar e mínimo.
# Índices base 0 nas 17 colunas dos CSVs, após remover cabeçalho.
DEDOS = {"polegar": 4, "indicador": 7, "medio": 10, "anelar": 13, "minimo": 16}
TAREFAS = {1: "Indicador", 2: "Medio", 3: "Anelar-Minimo", 4: "Polegar",
           5: "Pinca-Indicador", 6: "Pinca-Medio", 7: "Tripode", 8: "Cinco-dedos"}
