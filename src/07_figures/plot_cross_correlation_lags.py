"""Figura 8: correlação RMS–ângulo por lag para T1 e T5 (EDC/FDS).

Lê a tabela agregada do Experimento 20 com colunas tarefa, regiao,
cadencia_hz, lag_s, r_mediana_sujeitos. Não recalcula correlações.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2166ac"
ORANGE = "#e67e22"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True, help="20_correlacao_cruzada_por_tarefa.csv")
    p.add_argument("--saida", type=Path, required=True, help="Pasta para PDF e PNG")
    args = p.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    required = {"tarefa", "regiao", "cadencia_hz", "lag_s", "r_mediana_sujeitos"}
    if missing := required - set(df.columns):
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    df["regiao"] = df["regiao"].astype(str).str.upper()
    for c in ("tarefa", "cadencia_hz", "lag_s", "r_mediana_sujeitos"):
        df[c] = pd.to_numeric(df[c], errors="raise")
    if not np.isfinite(df[["lag_s", "r_mediana_sujeitos"]].to_numpy()).all():
        raise ValueError("CSV contém lags ou correlações inválidas")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True, sharey=True, layout="constrained")
    for ax, (task, reg), panel in zip(axs.flat,
                                      ((1, "EDC"), (1, "FDS"), (5, "EDC"), (5, "FDS")),
                                      ("(a)", "(b)", "(c)", "(d)")):
        selection = df[df["tarefa"].eq(task) & df["regiao"].eq(reg)]
        for cad, color, marker in ((0.50, BLUE, "o"), (0.75, ORANGE, "s")):
            curve = selection[np.isclose(selection["cadencia_hz"], cad)].sort_values("lag_s")
            if curve.empty:
                raise ValueError(f"Faltam dados para T{task}, {reg}, {cad:.2f} Hz")
            if curve["lag_s"].duplicated().any():
                raise ValueError(f"Lags duplicados em T{task}, {reg}, {cad:.2f} Hz")
            ax.plot(curve["lag_s"], curve["r_mediana_sujeitos"], "-",
                    color=color, marker=marker, markersize=3.4,
                    linewidth=1.5, label=f"{cad:.2f} Hz")
            print(f"T{task} {reg} {cad:.2f} Hz: {len(curve)} lags, "
                  f"intervalo [{curve['lag_s'].min():.2f}, {curve['lag_s'].max():.2f}] s")
        ax.axhline(0, linewidth=.8, color="gray")
        ax.axvline(0, linewidth=.8, linestyle="--", color="gray")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=.2)
        ax.text(.02, 1.03, f"{panel} T{task} · {reg}", transform=ax.transAxes,
                ha="left", va="bottom", weight="bold")
        ax.set_xlabel("Lag τ (s)")
        ax.set_ylabel("Median Pearson correlation")
    axs[0, 0].legend(loc="best", frameon=False, fontsize=8)
    args.saida.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = args.saida / f"fig08_cross_correlation_lags.{suffix}"
        fig.savefig(path, dpi=400 if suffix == "png" else None, bbox_inches="tight")
        print("Criado:", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
