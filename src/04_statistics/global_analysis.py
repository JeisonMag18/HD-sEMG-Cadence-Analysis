
"""
Análise global das características extraídas dos sinais HD-sEMG.

Agrega RMS, ARV, MNF e MDF por gravação, participante,
tarefa e frequência de movimento.

Calcula também correlações exploratórias entre o RMS
e a cinemática do dedo indicador.

Os dados são carregados dos arquivos de cache (.npz),
sem necessidade de reler os sinais EMG originais.

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Permite importar os módulos compartilhados de src/02_features.
DIRETORIO_FEATURES = Path(__file__).resolve().parents[1] / "02_features"
sys.path.insert(0, str(DIRETORIO_FEATURES))

import config as C
from features import ler_cache, ler_exclusoes



def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10 or np.std(x[mask]) < 1e-9 or np.std(y[mask]) < 1e-9:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def main():
    arquivos = sorted(C.CACHE.glob("*.npz"))
    excluidas = ler_exclusoes()
    linhas = []
    for idx, arquivo in enumerate(arquivos, 1):
        id_ = arquivo.stem
        if id_ in excluidas or id_.split("__")[-1] in excluidas:
            continue
        d = ler_cache(arquivo)
        if not all(k in d for k in ("rms", "arv", "mnf", "mdf", "angulos", "tempo_s")):
            print(f"Ignorando arquivo incompatível {arquivo.name}")
            continue
        rms, arv, mnf, mdf = (d[n] for n in ("rms", "arv", "mnf", "mdf"))
        t = d["tempo_s"]
        sujeito = id_.split("__")[0]
        tarefa = int(d["tarefa"]) if "tarefa" in d else -1
        frequencia = float(d["frequencia"]) if "frequencia" in d else np.nan
        linha = {"id": id_, "sujeito": sujeito, "tarefa": tarefa, "nome_tarefa": C.TAREFAS.get(tarefa, "?"),
                 "freq_hz": round(frequencia, 2), "n_janelas": len(t)}
        for sig_nome, matriz in (("rms",rms), ("arv",arv), ("mnf",mnf), ("mdf",mdf)):
            for musculo, fatia in (("EDC",slice(0,64)), ("FDS",slice(64,128))):
                v = matriz[:, fatia]
                linha[f"{sig_nome}_{musculo}_media"] = float(np.nanmean(v))
                linha[f"{sig_nome}_{musculo}_variabilidade_espacial"] = float(np.nanmean(np.nanstd(v, axis=1)))
        ang = d["angulos"][:, 1]
        vel = np.gradient(ang, t)
        for musculo, fatia in (("EDC", slice(0,64)), ("FDS",slice(64,128))):
            sinal = rms[:, fatia].mean(axis=1)
            linha[f"corr_RMS_{musculo}_angulo_indicador"] = pearson(sinal, ang)
            linha[f"corr_RMS_{musculo}_velocidade_indicador"] = pearson(sinal, vel)
        linhas.append(linha)
        if idx % 100 == 0:
            print(f"Analisados {idx}/{len(arquivos)} caches", flush=True)
    if not linhas:
        raise FileNotFoundError(
            "Nenhuma característica válida encontrada. "
            "Execute extract_features.py e confira o diretório do cache."
        )
    df = pd.DataFrame(linhas)
    df.to_csv(C.RESULTADOS / "03_resumo_por_gravacao.csv", index=False, encoding="utf-8-sig")
    por_pessoa = df.groupby(["sujeito", "tarefa", "freq_hz"], as_index=False).mean(numeric_only=True)
    por_pessoa.to_csv(C.RESULTADOS / "03_resumo_por_sujeito_tarefa_freq.csv", index=False, encoding="utf-8-sig")

    por_condicao = por_pessoa.groupby(
        ["tarefa", "freq_hz"], as_index=False
    ).agg(
        n_sujeitos=("sujeito", "nunique"),

        rms_EDC_media=("rms_EDC_media", "mean"),
        rms_FDS_media=("rms_FDS_media", "mean"),

        arv_EDC_media=("arv_EDC_media", "mean"),
        arv_FDS_media=("arv_FDS_media", "mean"),

        mnf_EDC_media=("mnf_EDC_media", "mean"),
        mnf_FDS_media=("mnf_FDS_media", "mean"),

        mdf_EDC_media=("mdf_EDC_media", "mean"),
        mdf_FDS_media=("mdf_FDS_media", "mean")
    )

    por_condicao.to_csv(C.RESULTADOS / "03_resumo_condicoes.csv", index=False, encoding="utf-8-sig")

    # Visual: média por sujeito/condição (não tratar milhares de janelas como sujeitos independentes).
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    for ax, musculo in zip(axs, ("EDC", "FDS")):
        largura = 0.35
        x = np.arange(1,9)
        for j, freq in enumerate((0.5, 0.75)):
            sub = por_condicao[np.isclose(por_condicao["freq_hz"],freq)]
            vals = [float(sub.loc[sub["tarefa"]==t,f"rms_{musculo}_media"].iloc[0])
                    if (sub["tarefa"]==t).any() else np.nan for t in x]
            ax.bar(x + (j - 0.5)*largura, vals, largura, label=f"{freq} Hz")
        ax.set_xticks(x)
        ax.set_xticklabels([C.TAREFAS[int(t)] for t in x], rotation=20, ha="right")
        ax.set_ylabel("RMS (unidade do CSV)")
        ax.set_title(f"RMS global {musculo} — média entre participantes por condição")
        ax.legend()
    destino = C.FIGURAS / "03_rms_por_tarefa.png"
    fig.savefig(destino, dpi=170)
    plt.close(fig)
    print("Gravações analisadas:", len(df), "| Sujeitos:", df.sujeito.nunique())
    print("Tabelas e figura em:", C.RESULTADOS)
    print("IMPORTANTE: rever Kinematic_review.ods antes da análise final.")


if __name__ == "__main__":
    main()
