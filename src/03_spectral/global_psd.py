"""PSD global HD-sEMG (IA753): todas as gravações em cache, 21 participantes.

Uso: python global_psd.py

Agregação hierárquica e balanceada:
  64 canais -> gravação -> média das repetições por sujeito/tarefa/condição
  -> média das tarefas por sujeito/condição -> média dos sujeitos.

Welch é calculado sobre 5--40 s de EMG bruto (não sobre o RMS), com segmentos
Hann de ~1 s e 50% de sobreposição. Mantém os dados de entrada intactos.
A análise global de PSD não é idêntica às MNF/MDF derivadas de janelas móveis
individuais de 2 s; essa distinção deve constar no artigo.
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.integrate import trapezoid

import config as C
from features import indexar_gravacoes, ler_csv_numerico, ler_exclusoes


def main():
    arquivos_cache = {p.stem for p in C.CACHE.glob("*.npz")}
    excluidas = ler_exclusoes()
    gravacoes, problemas = indexar_gravacoes()
    gravacoes = [
        g for g in gravacoes
        if g.id in arquivos_cache
        and g.id not in excluidas
        and g.emg.name not in excluidas
        and g.emg.stem not in excluidas
    ]
    print(f"Gravações com cache selecionadas: {len(gravacoes)}")
    print(f"Anomalias do inventário: {len(problemas)}")
    if not gravacoes:
        raise RuntimeError("Nenhuma gravação encontrada em C.CACHE")

    # [(sujeito, tarefa, freq_mov, musculo)] -> [(PSD_abs, PSD_norm), ...]
    por_repeticao = defaultdict(list)
    situacao = []
    freq_emg = None
    n_segmento = int(round(C.FS_EMG))
    inicio = int(round(C.INTERVALO_INICIO * C.FS_EMG))
    fim = int(round(C.INTERVALO_FIM * C.FS_EMG))

    for i, g in enumerate(gravacoes, 1):
        try:
            emg = ler_csv_numerico(g.emg, 128)
            sinal = emg[inicio:fim]
            if len(sinal) < 2 * n_segmento:
                raise ValueError("Intervalo EMG insuficiente")
            f, pxx = welch(
                sinal, fs=C.FS_EMG, window="hann",
                nperseg=n_segmento, noverlap=n_segmento // 2,
                detrend="constant", scaling="density", axis=0,
            )
            selecionar = (f >= C.FAIXA_ESPECTRAL[0]) & (f <= C.FAIXA_ESPECTRAL[1])
            fs = f[selecionar]
            if freq_emg is None:
                freq_emg = fs
            elif not np.array_equal(freq_emg, fs):
                raise ValueError("Grade de frequências incompatível")

            for musculo, canais in (("EDC", slice(0, 64)), ("FDS", slice(64, 128))):
                # Média dos periodogramas entre canais de um mesmo músculo
                psd_abs = pxx[selecionar, canais].mean(axis=1)
                potencia = trapezoid(psd_abs, fs)
                if not np.isfinite(potencia) or potencia <= 0:
                    raise ValueError(f"Potência inválida: {musculo}")
                psd_norm = psd_abs / potencia
                por_repeticao[(g.sujeito, g.tarefa, g.frequencia, musculo)].append(
                    (psd_abs.astype(np.float32), psd_norm.astype(np.float32))
                )
            situacao.append({"id": g.id, "status": "ok", "motivo": ""})
        except Exception as erro:
            situacao.append({"id": g.id, "status": "erro", "motivo": str(erro)})
            print(f"[ERRO] {g.id}: {erro}")
        if i % 50 == 0 or i == len(gravacoes):
            print(f"PSD processadas: {i}/{len(gravacoes)}", flush=True)

    pd.DataFrame(situacao).to_csv(
        C.RESULTADOS / "12_psd_processamento.csv", index=False, encoding="utf-8-sig"
    )
    if freq_emg is None:
        raise RuntimeError("Nenhuma PSD válida foi calculada")

    # Repetições -> uma curva por participante/tarefa/frequência/músculo.
    por_tarefa = {}
    for chave, curvas in por_repeticao.items():
        arr = np.asarray(curvas)  # [repeticoes, 2 (abs,norm), frequencias]
        por_tarefa[chave] = arr.mean(axis=0)

    # Tarefas -> uma curva por participante/frequência/músculo.
    por_sujeito = defaultdict(list)
    for (sujeito, tarefa, freq_mov, musculo), curva in por_tarefa.items():
        por_sujeito[(sujeito, freq_mov, musculo)].append(curva)
    por_sujeito = {chave: np.mean(curvas, axis=0) for chave, curvas in por_sujeito.items()}

    # Participantes -> médias e desvios-padrão entre participantes.
    linhas = {"frequencia_emg_hz": freq_emg}
    fig, eixos = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    resumo = []
    for i, musculo in enumerate(("EDC", "FDS")):
        for freq_mov in (0.5, 0.75):
            curvas = [
                curva for (sujeito, freq, m), curva in por_sujeito.items()
                if m == musculo and np.isclose(freq, freq_mov)
            ]
            if not curvas:
                continue
            dados = np.stack(curvas)  # [sujeitos, abs/norm, frequencias]
            n_sujeitos = len(curvas)
            tag = f"{musculo}_{str(freq_mov).replace('.', '')}"
            for k, tipo in enumerate(("absoluta", "normalizada")):
                media = dados[:, k, :].mean(axis=0)
                desvio = dados[:, k, :].std(axis=0, ddof=1) if n_sujeitos > 1 else np.zeros_like(media)
                linhas[f"psd_{tipo}_media_{tag}"] = media
                linhas[f"psd_{tipo}_dp_{tag}"] = desvio
                ax = eixos[i, k]
                ax.plot(freq_emg, media, label=f"{freq_mov:.2f} Hz (n={n_sujeitos})")
                # Sombras: desvio-padrão ENTRE participantes, não intervalo de confiança.
                if k == 0:
                    # No eixo log, média - DP pode ser negativa.
                    # Exibir só a média absoluta; DP é preservado no CSV.
                    ax.set_yscale("log")
                else:
                    baixo = np.maximum(media - desvio, 0)
                    alto = media + desvio
                    ax.fill_between(freq_emg, baixo, alto, alpha=0.12)
                ax.set_title(f"{musculo} — PSD {tipo}")
                ax.set_xlabel("Frequência do EMG (Hz)")
                ax.set_ylabel("PSD (unid. CSV²/Hz)" if k == 0 else "PSD normalizada (1/Hz)")
                ax.grid(alpha=0.2)
                ax.legend()
            resumo.append({
                "musculo": musculo,
                "freq_mov_hz": freq_mov,
                "n_sujeitos": n_sujeitos,
                "potencia_media_10_500": float(np.mean([
                    trapezoid(c[0], freq_emg) for c in curvas
                ])),
            })

    destino = C.FIGURAS / "12_psd_global_21_participantes.png"
    fig.savefig(destino, dpi=200)
    plt.close(fig)
    pd.DataFrame(linhas).to_csv(
        C.RESULTADOS / "12_psd_curvas_globais.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(resumo).to_csv(
        C.RESULTADOS / "12_psd_resumo.csv", index=False, encoding="utf-8-sig"
    )
    print("\nGravações com PSD válida:", sum(r["status"] == "ok" for r in situacao))
    print("Figura:", destino)
    print("Dados:", C.RESULTADOS / "12_psd_curvas_globais.csv")
    print("Resumo:", C.RESULTADOS / "12_psd_resumo.csv")
    print("Atenção: verificar qualidade cinemática e a interpretação fisiológica.")

if __name__ == "__main__":
    main()
