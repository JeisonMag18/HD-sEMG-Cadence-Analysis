"""Funções compartilhadas de leitura, registro, características e validação."""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.signal import welch
import config as C


@dataclass(frozen=True)
class Gravacao:
    sujeito: str
    tarefa: int
    frequencia: float
    repeticao: int
    emg: Path
    ang: Path

    @property
    def id(self) -> str:
        return f"{self.sujeito}__{self.emg.stem}"


_PADRAO = re.compile(r"^(Sub\d+)_(\d+)_([^_]+)_([^_]+)_(\d+)\.csv$", re.I)


def ler_frequencia(codigo: str) -> float:
    s = codigo.replace(",", ".")
    if s in ("05", "050", "0.5", "0.50", "50"):
        return 0.5
    if s in ("075", "0750", "0.75", "75"):
        return 0.75
    raise ValueError(f"Código de frequência desconhecido no nome de arquivo: {codigo!r}")


def indexar_gravacoes() -> tuple[list[Gravacao], list[dict]]:
    arquivos = sorted(C.DATASET.glob("Sub*/HD_sEMG/*.csv"))
    gravacoes, problemas = [], []
    for emg in arquivos:
        m = _PADRAO.match(emg.name)
        if m is None:
            # Rest/start_position não entram nas tarefas 1..8.
            if "rest" not in emg.stem.lower() and "start" not in emg.stem.lower():
                problemas.append({"arquivo": str(emg), "motivo": "nome não reconhecido"})
            continue
        sujeito, tarefa, codigo_freq, duracao, repeticao = m.groups()
        try:
            tarefa = int(tarefa)
            frequencia = ler_frequencia(codigo_freq)
            repeticao = int(repeticao)
        except ValueError as e:
            problemas.append({"arquivo": str(emg), "motivo": str(e)})
            continue
        if tarefa not in C.TAREFAS:
            problemas.append({"arquivo": str(emg), "motivo": "tarefa fora de 1..8"})
            continue
        ang = emg.parent.parent / "HandKinematics" / "Angles" / emg.name
        if not ang.exists():
            problemas.append({"arquivo": str(emg), "motivo": "ângulos não encontrados"})
            continue
        gravacoes.append(Gravacao(sujeito, tarefa, frequencia, repeticao, emg, ang))
    return gravacoes, problemas


def ler_exclusoes() -> set[str]:
    caminho = C.PROJETO / "exclusoes.csv"
    if not caminho.exists():
        caminho.write_text("arquivo,motivo\n", encoding="utf-8")
        return set()
    df = pd.read_csv(caminho, dtype=str).fillna("")
    if "arquivo" not in df.columns:
        raise ValueError("exclusoes.csv precisa da coluna 'arquivo'")
    return {x.strip() for x in df["arquivo"].tolist() if x.strip()}


def primeira_linha_numerica(caminho: Path, n_colunas: int) -> int:
    """Reconhece cabeçalhos livres sem 'pular 5' para todo arquivo."""
    with caminho.open("r", encoding="utf-8-sig", errors="replace", newline="") as arq:
        for num, linha in enumerate(csv.reader(arq)):
            if num > 60:
                break
            while linha and not linha[-1].strip():
                linha.pop()
            if len(linha) != n_colunas:
                continue
            try:
                arr = np.asarray(
                    [float(x) if x.strip() else np.nan for x in linha],
                    dtype=float
                )
            except ValueError:
                continue

            # Para a cinemática, aceitar a primeira linha de dados
            # mesmo que alguns ângulos estejam ausentes.
            if n_colunas == 17:
                if np.all(np.isfinite(arr[:2])) and np.isfinite(arr).sum() >= 5:
                    return num
            else:
                if np.all(np.isfinite(arr)):
                    return num
    raise ValueError(f"Cabeçalho não reconhecido: {caminho}; esperado {n_colunas} colunas numéricas")


def ler_csv_numerico(
    caminho: Path,
    n_colunas: int,
    permitir_nan_fora_intervalo: bool = False
) -> np.ndarray:
    inicio = primeira_linha_numerica(caminho, n_colunas)
    # Um arquivo por vez. Dados em float32 para limitar memória RAM.
    tabela = pd.read_csv(caminho, skiprows=inicio, header=None, dtype=np.float32,
                         low_memory=False)
    # Alguns gravadores acrescentam vírgula extra no final da linha.
    if tabela.shape[1] == n_colunas + 1 and tabela.iloc[:, -1].isna().all():
        tabela = tabela.iloc[:, :-1]
    if tabela.shape[1] != n_colunas:
        raise ValueError(f"{caminho.name}: esperadas {n_colunas} colunas, recebidas {tabela.shape[1]}")
    dados = tabela.to_numpy(dtype=np.float32, copy=True)
    # Não remover linhas internas: isso destruiria a sincronização.
    if not np.all(np.isfinite(dados)):

        if not permitir_nan_fora_intervalo:
            raise ValueError(
                f"{caminho.name}: NaN/inf detectado"
            )

        # Verificar todo o intervalo cinemático utilizado.
        inicio = int(np.floor(C.INTERVALO_INICIO * C.FS_ANG))
        fim = int(np.floor(C.INTERVALO_FIM * C.FS_ANG)) + 1

        if len(dados) < fim:
            raise ValueError("Cinemática mais curta que o intervalo esperado")

        if not np.all(np.isfinite(dados[inicio:fim, :])):
            raise ValueError(
                f"{caminho.name}: NaN/inf dentro de 5–40 s"
            )
    return dados


def _mediana_espectral(pxx: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    cumul = np.cumsum(pxx, axis=0)
    total = cumul[-1]
    k = np.argmax(cumul >= (total[None, :] * 0.5), axis=0)
    mediana = freqs[k].astype(np.float64)
    mediana[total <= 1e-20] = np.nan
    return mediana



def extrair_caracteristicas(
    emg: np.ndarray,
    ang: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Extrai ARV e RMS com janelas centradas de 300 ms.

    MNF e MDF continuam utilizando janelas causais
    de 2 s, com PSD estimada pelo metodo de Welch.

    Passo entre estimativas: 250 ms.
    """

    if (emg.ndim != 2 or emg.shape[1] != 128
            or ang.ndim != 2 or ang.shape[1] != 17):
        raise ValueError(
            "Formatos esperados: EMG (N,128), angulos (M,17)"
        )

    fim = min(
        C.INTERVALO_FIM,
        (len(emg) - 1) / C.FS_EMG,
        (len(ang) - 1) / C.FS_ANG
        - C.DESLOCAMENTO_ANGULOS
    )

    inicio = (
        C.INTERVALO_INICIO
        + max(C.JANELA_PSD, C.JANELA_RMS)
    )

    tempos = np.arange(
        inicio, fim + 1e-8, C.PASSO, dtype=float
    )

    if len(tempos) < 5:
        raise ValueError("Tempo util insuficiente")

    # RMS CENTRADO: N = 2M + 1
    M = int(round(C.JANELA_RMS * C.FS_EMG / 2))
    n_amp = 2 * M + 1

    # Welch: manter janela espectral anterior
    n_psd = int(round(C.JANELA_PSD * C.FS_EMG))
    nperseg_welch = int(round(1.0 * C.FS_EMG))

    lista = {
        nome: []
        for nome in ("arv", "rms", "mnf", "mdf")
    }

    tempos_validos = []

    for t in tempos:

        i_centro = int(round(t * C.FS_EMG))

        # RMS e ARV: janela centrada no instante t
        amp = emg[
            i_centro - M : i_centro + M + 1, :
        ]

        # PSD: janela nos 2 segundos anteriores
        bloco = emg[
            i_centro - n_psd : i_centro, :
        ]

        if (amp.shape != (n_amp, 128)
                or bloco.shape != (n_psd, 128)):
            continue

        # ARV
        arv = np.mean(
            np.abs(amp),
            axis=0,
            dtype=np.float64
        )

        # RMS MOVEL CENTRADO
        rms = np.sqrt(
            np.mean(
                amp.astype(np.float64)**2,
                axis=0
            )
        )

        lista["arv"].append(arv)
        lista["rms"].append(rms)

        # PSD PELO METODO DE WELCH
        f, pxx = welch(
            bloco,
            fs=C.FS_EMG,
            window="hann",
            axis=0,
            nperseg=nperseg_welch,
            noverlap=nperseg_welch // 2,
            detrend="constant",
            scaling="density"
        )

        mascara = (
            (f >= C.FAIXA_ESPECTRAL[0])
            & (f <= C.FAIXA_ESPECTRAL[1])
        )

        f = f[mascara]
        pxx = pxx[mascara, :]

        total = pxx.sum(
            axis=0, dtype=np.float64
        )

        # FREQUENCIA MEDIA
        mnf = (
            (f[:, None] * pxx).sum(
                axis=0, dtype=np.float64
            ) / np.maximum(total, 1e-20)
        )

        mnf[total <= 1e-20] = np.nan

        # FREQUENCIA MEDIANA
        mdf = _mediana_espectral(pxx, f)

        lista["mnf"].append(mnf)
        lista["mdf"].append(mdf)

        tempos_validos.append(t)

    tempos_validos = np.asarray(
        tempos_validos,
        dtype=np.float32
    )

    tempos_ang = (
        np.arange(len(ang), dtype=float)
        / C.FS_ANG
    )

    ang_5 = np.column_stack([
        np.interp(
            tempos_validos + C.DESLOCAMENTO_ANGULOS,
            tempos_ang,
            ang[:, c]
        )
        for c in C.DEDOS.values()
    ]).astype(np.float32)

    return {
        "tempo_s": tempos_validos,
        "angulos": ang_5,
        **{
            nome: np.asarray(
                valores, dtype=np.float32
            )
            for nome, valores in lista.items()
        }
    }


def ler_cache(caminho: Path) -> dict[str, np.ndarray]:
    with np.load(caminho, allow_pickle=False) as dados:
        return {k: dados[k].copy() for k in dados.files}


def mapear_8x8(valores: np.ndarray, mapa, musculo: str) -> tuple[np.ndarray, bool]:
    if mapa is None:
        return valores.reshape(8, 8), False
    mapa = np.asarray(mapa)
    if mapa.shape != (8, 8) or set(mapa.flatten().tolist()) != set(range(64)):
        raise ValueError(f"MAPA_{musculo} deve ter exatamente os índices 0..63 em grade 8x8")
    return valores[mapa], True
