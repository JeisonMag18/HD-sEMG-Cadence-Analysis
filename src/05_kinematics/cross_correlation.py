"""
Análise descritiva da correlação cruzada entre RMS e cinemática.

Utiliza os arquivos NPZ gerados na etapa de extração,
sem reler os sinais brutos.

Convenção:
    r(tau) = corr(RMS(t), angulo(t + tau))

Uma defasagem positiva indica que o ângulo foi avaliado
em um instante posterior ao RMS.

Devido à periodicidade dos movimentos, o máximo da
correlação não deve ser interpretado isoladamente como
uma estimativa de atraso fisiológico.

"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Localização dos módulos compartilhados.
DIRETORIO_FEATURES = (
    Path(__file__).resolve().parents[1] / "02_features"
)

sys.path.insert(0, str(DIRETORIO_FEATURES))

import config as C
from features import ler_exclusoes

# As colunas de ângulos foram selecionadas durante
# a extração de características em features.py.
DEDOS = list(C.DEDOS.keys())
# Dedo de referência por tarefa: nas tarefas multiarticulares é uma ESCOLHA
# operacional e não indica que os outros dedos não participam.
ALVO_POR_TAREFA = {
    1: "indicador", 2: "medio", 3: "anelar", 4: "polegar",
    5: "indicador", 6: "medio", 7: "indicador", 8: "indicador",
}


def pearson_lag(x, y, passos):
    """r entre x[n] e y[n+passos], com amostras pareadas válidas."""
    if passos > 0:
        xx, yy = x[:-passos], y[passos:]
    elif passos < 0:
        xx, yy = x[-passos:], y[:passos]
    else:
        xx, yy = x, y
    mask = np.isfinite(xx) & np.isfinite(yy)
    if mask.sum() < 20:
        return np.nan
    xx, yy = xx[mask].astype(float), yy[mask].astype(float)
    if np.std(xx) <= 1e-10 or np.std(yy) <= 1e-10:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


def main():
    cache_analise = C.RESULTADOS / "features_welch_centrado"
    excluidas = ler_exclusoes()

    arquivos = [
        caminho
        for caminho in sorted(cache_analise.glob("*.npz"))
        if caminho.stem not in excluidas
           and caminho.name not in excluidas
    ]
    if not arquivos:
        raise RuntimeError(f"Nenhum cache NPZ em {cache_analise}")
    # 0,25 s corresponde à resolução temporal das características já extraídas.
    passo = float(C.PASSO)
    if not np.isclose(passo, 0.25):
        print(f"Aviso: passo temporal configurado = {passo} s")
    # Lags até ±1,0 s, mantendo a periodicidade da tarefa explícita na discussão.
    ks = np.arange(-int(round(1.0 / passo)), int(round(1.0 / passo)) + 1)
    linhas = []
    ignorados = []


    for i, caminho in enumerate(arquivos, start=1):
        linhas_gravacao = []

        try:
            with np.load(caminho, allow_pickle=False) as d:
                rms = d["rms"]
                ang = d["angulos"]
                t = d["tempo_s"]
                tarefa = int(d["tarefa"])
                frequencia = float(d["frequencia"])

            if (
                rms.ndim != 2
                or rms.shape[1] != 128
                or ang.shape != (len(rms), len(DEDOS))
            ):
                raise ValueError(
                    "Dimensões RMS/ângulos incompatíveis"
                )

            if len(t) != len(rms) or len(t) < 30:
                raise ValueError(
                    "Tempo incompatível ou registro muito curto"
                )

            if not np.allclose(
                np.diff(t), passo, atol=1e-3
            ):
                raise ValueError("Tempo não uniforme")

            dedo = ALVO_POR_TAREFA[tarefa]
            y = ang[:, DEDOS.index(dedo)]
            sujeito = caminho.stem.split("__")[0]

            for musculo, fatia in (
                ("EDC", slice(0, 64)),
                ("FDS", slice(64, 128)),
            ):
                x = np.nanmean(rms[:, fatia], axis=1)

                for k in ks:
                    linhas_gravacao.append({
                        "id": caminho.stem,
                        "sujeito": sujeito,
                        "tarefa": tarefa,
                        "freq_mov_hz": round(frequencia, 2),
                        "dedo_referencia": dedo,
                        "musculo": musculo,
                        "lag_s": round(float(k * passo), 3),
                        "r": pearson_lag(x, y, int(k)),
                    })

            # Inclui os resultados apenas após o
            # processamento completo da gravação.
            linhas.extend(linhas_gravacao)

        except (KeyError, ValueError, IndexError) as exc:
            ignorados.append({
                "arquivo": caminho.name,
                "motivo": str(exc),
            })

        if i % 200 == 0:
            print(
                f"Lidos {i}/{len(arquivos)} caches",
                flush=True,
            )


    df = pd.DataFrame(linhas)
    if df.empty:
        raise RuntimeError("Nenhuma correlação válida calculada")
    saida = Path(C.RESULTADOS)
    figs = Path(C.FIGURAS)
    saida.mkdir(exist_ok=True, parents=True)
    figs.mkdir(exist_ok=True, parents=True)
    df.to_csv(saida / "13_corr_cruzada_por_gravacao.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ignorados, columns=["arquivo", "motivo"]).to_csv(
        saida / "13_corr_cruzada_ignorados.csv", index=False, encoding="utf-8-sig"
    )

    # Primeiro resumir repetições de uma pessoa numa condição; depois entre sujeitos.
    por_sujeito = (df.groupby(
        ["sujeito", "tarefa", "freq_mov_hz", "dedo_referencia", "musculo", "lag_s"],
        as_index=False, dropna=False
    ).agg(r_mediano=("r", "median"), n_gravacoes=("id", "nunique")))
    por_sujeito.to_csv(saida / "13_corr_cruzada_por_sujeito.csv", index=False, encoding="utf-8-sig")
    global_ = (por_sujeito.groupby(
        ["tarefa", "freq_mov_hz", "dedo_referencia", "musculo", "lag_s"],
        as_index=False, dropna=False
    ).agg(r_mediana_sujeitos=("r_mediano", "median"), n_sujeitos=("r_mediano", "count")))
    global_.to_csv(saida / "13_corr_cruzada_global.csv", index=False, encoding="utf-8-sig")

    # Ilustração das tarefas T1 e T5, considerando os
    # participantes com gravações válidas em cada condição.
    fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, layout="constrained")
    for linha, tarefa in enumerate((1, 5)):
        for coluna, musculo in enumerate(("EDC", "FDS")):
            ax = axs[linha, coluna]
            sub = global_[(global_["tarefa"] == tarefa) & (global_["musculo"] == musculo)]
            for freq in (0.50, 0.75):
                sf = sub[np.isclose(sub["freq_mov_hz"], freq)].sort_values("lag_s")
                ax.plot(sf["lag_s"], sf["r_mediana_sujeitos"], marker="o", label=f"{freq:.2f} Hz")
            ax.axhline(0, color="gray", lw=0.8)
            ax.axvline(0, color="gray", lw=0.8, ls="--")
            ax.set_ylim(-1.05, 1.05)
            ax.set_title(f"T{tarefa} ({ALVO_POR_TAREFA[tarefa]}) — {musculo}")
            ax.set_ylabel("Mediana de r entre participantes")
            ax.grid(alpha=0.25)
            ax.legend()
    axs[1, 0].set_xlabel("Defasagem τ (s): ângulo em t + τ")
    axs[1, 1].set_xlabel("Defasagem τ (s): ângulo em t + τ")
    caminho_fig = figs / "13_correlacao_cruzada_global_T1_T5.png"
    fig.savefig(caminho_fig, dpi=200)
    plt.close(fig)
    print("Gravações distintas analisadas:", df["id"].nunique())
    print("Participantes:", df["sujeito"].nunique())
    print("Arquivos ignorados:", len(ignorados))
    print("Resumo:", saida / "13_corr_cruzada_global.csv")
    print("Figura:", caminho_fig)
    print("ATENÇÃO: valores máximos e lags não demonstram atraso eletromecânico; movimentos são periódicos.")


if __name__ == "__main__":
    main()
