
"""Figura 6: CV do RMS entre 64 canais, por tarefa, região e cadência.

Entrada: CSV de descritores por gravação, contendo
sujeito, tarefa, cadencia_hz, regiao, descritor, valor.
Agrupa primeiro repetições dentro de cada participante; os pontos são
médias dos participantes e as barras mostram ±1 desvio-padrão.

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
DARK = "#243144"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True, help="20_descritores_por_gravacao.csv")
    p.add_argument("--comparacoes", type=Path, default=None,
                   help="20_cv_comparacoes_por_tarefa.csv (opcional)")
    p.add_argument("--saida", type=Path, required=True, help="Pasta para PDF e PNG")
    args = p.parse_args()

    raw = pd.read_csv(args.csv, encoding="utf-8-sig")
    required = {"sujeito", "tarefa", "cadencia_hz", "regiao", "descritor", "valor"}
    if missing := required - set(raw.columns):
        raise ValueError(f"Colunas ausentes no CSV: {sorted(missing)}")

    raw = raw.loc[raw["descritor"].astype(str).str.upper().eq("CV_RMS")].copy()
    if raw.empty:
        raise ValueError("O CSV não contém linhas com descritor = CV_RMS")
    raw["regiao"] = raw["regiao"].astype(str).str.upper()
    raw["cadencia_hz"] = pd.to_numeric(raw["cadencia_hz"], errors="raise")
    raw["tarefa"] = pd.to_numeric(raw["tarefa"], errors="raise").astype(int)
    raw["valor"] = pd.to_numeric(raw["valor"], errors="raise")
    if not np.isfinite(raw["valor"]).all():
        raise ValueError("CV_RMS contém NaN ou infinito")

    subj = raw.groupby(["sujeito", "tarefa", "cadencia_hz", "regiao"], as_index=False)["valor"].mean()
    comp = pd.DataFrame()
    if args.comparacoes is not None:
        comp = pd.read_csv(args.comparacoes, encoding="utf-8-sig")
        necessary = {"regiao", "tarefa", "p_holm_16"}
        if missing := necessary - set(comp.columns):
            raise ValueError(f"Colunas ausentes nas comparações: {sorted(missing)}")
        comp["regiao"] = comp["regiao"].astype(str).str.upper()
        comp["tarefa"] = pd.to_numeric(comp["tarefa"], errors="raise").astype(int)
        comp["p_holm_16"] = pd.to_numeric(comp["p_holm_16"], errors="raise")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axs = plt.subplots(2, 1, figsize=(5.0, 5.4), sharex=True, layout="constrained")
    for ax, reg, panel in zip(axs, ("EDC", "FDS"), ("(a)", "(b)")):
        region = subj[subj["regiao"].eq(reg)]
        if region.empty:
            raise ValueError(f"Nenhum dado para {reg}")
        for cad, color, marker in ((0.50, BLUE, "o"), (0.75, ORANGE, "s")):
            data = region[np.isclose(region["cadencia_hz"], cad)]
            stats = data.groupby("tarefa")["valor"].agg(["mean", "std", "count"]).reindex(range(1, 9))
            if stats[["mean", "std"]].isna().any().any():
                raise ValueError(f"Dados insuficientes para {reg}, {cad:.2f} Hz, tarefas T1–T8")
            print(f"{reg} {cad:.2f} Hz: participantes por tarefa = {stats['count'].astype(int).to_list()}")
            ax.errorbar(range(1, 9), stats["mean"], yerr=stats["std"],
                        color=color, marker=marker, markersize=4.2,
                        linewidth=1.5, capsize=2.5, label=f"{cad:.2f} Hz")
        ax.set_ylabel("64-channel RMS CV (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.text(0.01, 1.03, f"{panel} {reg}", transform=ax.transAxes,
                weight="bold", ha="left", va="bottom")
        if not comp.empty:
            for t in comp.loc[(comp["regiao"].eq(reg)) & (comp["p_holm_16"] < .05), "tarefa"]:
                t = int(t)
                if 1 <= t <= 8:
                    vals = region.loc[region["tarefa"].eq(t), "valor"]
                    if not vals.empty:
                        # Annotation placed above the maximum of mean ± SD at this task.
                        grouped = region[region["tarefa"].eq(t)].groupby("cadencia_hz")["valor"].agg(["mean", "std"])
                        top = float((grouped["mean"] + grouped["std"]).max())
                        ax.annotate("*", (t, top), xytext=(0, 4),
                                    textcoords="offset points", ha="center", va="bottom", fontsize=10)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axs[-1].set_xticks(range(1, 9), [f"T{i}" for i in range(1, 9)])
    axs[-1].set_xlabel("Movement task")
    args.saida.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = args.saida / f"fig06_rms_cv.{suffix}"
        fig.savefig(path, dpi=400 if suffix == "png" else None, bbox_inches="tight")
        print("Criado:", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
