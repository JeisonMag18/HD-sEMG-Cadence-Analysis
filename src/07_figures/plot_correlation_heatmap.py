"""
Gera o mapa de calor das correlações RMS–ângulo em tau = 0.

Compara oito tarefas (T1–T8), duas regiões musculares
(EDC e FDS) e duas cadências (0.50 e 0.75 Hz).

Os coeficientes são carregados de um CSV previamente
calculado. Exporta a figura em PDF e PNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def bloco(df: pd.DataFrame, cad: float) -> np.ndarray:
    linhas = []
    for tarefa in range(1, 9):
        sub = df.loc[(df["tarefa"] == tarefa) & np.isclose(df["cadencia_hz"], cad)]
        linha = []
        for regiao in ("EDC", "FDS"):
            valores = sub.loc[sub["regiao"].astype(str).str.strip().str.upper() == regiao,
                              "r_mediana_sujeitos"]
            linha.append(float(valores.iloc[0]) if not valores.empty else np.nan)
        linhas.append(linha)
    return np.asarray(linhas, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path,
                        default=Path("resultados_ia753/20_final_centrado/20_correlacao_tau_zero.csv"))
    parser.add_argument("--saida", type=Path,
                        default=Path("resultados_ia753/31_figuras_publicacao"))
    args = parser.parse_args()

    if not args.csv.is_file():
        raise FileNotFoundError(
            f"CSV não encontrado: {args.csv}\n"
            "Use --csv CAMINHO_DO_ARQUIVO para indicar onde está seu CSV."
        )

    df = pd.read_csv(args.csv)
    obrigatorias = {"tarefa", "cadencia_hz", "regiao", "r_mediana_sujeitos"}
    faltantes = obrigatorias - set(df.columns)
    if faltantes:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(faltantes)}")

    mat = np.column_stack([bloco(df, 0.50), bloco(df, 0.75)])
    if np.isnan(mat).any():
        faltam = [(f"T{i+1}", ("0.50 EDC", "0.50 FDS", "0.75 EDC", "0.75 FDS")[j])
                  for i, j in zip(*np.where(np.isnan(mat)))]
        raise ValueError(f"Faltam valores para células do heatmap: {faltam}")

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })

    fig = plt.figure(figsize=(4.65, 6.35))
    # Posicoes fixas: elimina faixa branca excessiva acima do mapa.
    ax = fig.add_axes([0.11, 0.065, 0.70, 0.81])
    cmap = plt.get_cmap("RdBu_r")
    norm = Normalize(vmin=-1.0, vmax=1.0)
    im = ax.imshow(mat, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")

    ax.set_xticks(range(4), labels=["EDC", "FDS", "EDC", "FDS"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(8), labels=[f"T{i}" for i in range(1, 9)])
    ax.tick_params(axis="both", which="major", length=0, pad=5, labelsize=10.5)
    ax.tick_params(axis="both", which="minor", length=0)

    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.axvline(1.5, color="white", linewidth=2.3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Cabeçalhos por cadência, acima de cada par de colunas.
    for x, label in ((0.5, "0.50 Hz"), (2.5, "0.75 Hz")):
        ax.text(x, 1.105, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=11.5, clip_on=False)

    # Regra explícita para evitar preto sobre vermelho:
    # os valores positivos >= 0.40 são brancos; os negativos <= -0.55 também.
    # Valores próximos de zero permanecem pretos sobre fundos claros.
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            valor = float(mat[i, j])
            cor = "white" if (valor >= 0.40 or valor <= -0.55) else "black"
            ax.text(j, i, f"{valor:+.2f}", ha="center", va="center",
                    fontsize=10.3, fontweight="medium", color=cor)

    cax = fig.add_axes([0.84, 0.09, 0.047, 0.76])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks(np.arange(-1, 1.01, 0.25))
    cbar.ax.tick_params(labelsize=10)

    args.saida.mkdir(parents=True, exist_ok=True)
    base = args.saida / "34_correlacao_heatmap_publicacao"
    fig.savefig(base.with_suffix(".pdf"), pad_inches=0.02)
    fig.savefig(base.with_suffix(".png"), dpi=600, pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] PDF: {base.with_suffix('.pdf')}")
    print(f"[OK] PNG: {base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
