"""Extrai características de TODO o dataset, 1 gravação por vez, com cache.

Teste: python 01_extrair_tudo.py --limite 1
Um sujeito: python 01_extrair_tudo.py --sujeito Sub001
Completo: python 01_extrair_tudo.py --todos
Retoma sem recalcular arquivos NPZ existentes. Use --refazer para substituir.
"""
import argparse
import gc
import time
from pathlib import Path
import numpy as np
import pandas as pd
import config as C
CACHE_CENTRADO = C.RESULTADOS / "features_welch_centrado"
CACHE_CENTRADO.mkdir(parents=True, exist_ok=True)
from funcoes import indexar_gravacoes, ler_exclusoes, ler_csv_numerico, extrair_caracteristicas


def gravar_status(status: list[dict]):
    pd.DataFrame(status, columns=["id", "status", "janelas", "duracao_s",
                                   "motivo"]).to_csv(
        C.RESULTADOS / "01_processamento_centrado.csv", index=False, encoding="utf-8-sig")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--todos", action="store_true", help="Processar todas as gravações elegíveis encontradas no dataset")
    a.add_argument("--limite", type=int, default=1, help="Inicialmente 1 gravação; --todos remove o limite")
    a.add_argument("--sujeito", type=str, help="Ex.: Sub001")
    a.add_argument("--tarefa", type=int, choices=range(1, 9))
    a.add_argument("--refazer", action="store_true")
    args = a.parse_args()
    gravacoes, problemas = indexar_gravacoes()
    if not gravacoes:
        raise FileNotFoundError(f"Sem pares HD_sEMG/Angles em {C.DATASET}")
    excluidas = ler_exclusoes()
    gravacoes = [g for g in gravacoes if g.emg.name not in excluidas and g.id not in excluidas]
    if args.sujeito:
        gravacoes = [g for g in gravacoes if g.sujeito.lower() == args.sujeito.lower()]
    if args.tarefa:
        gravacoes = [g for g in gravacoes if g.tarefa == args.tarefa]
    if not args.todos:
        gravacoes = gravacoes[:max(1, args.limite)]
    print(f"Pares selecionados: {len(gravacoes)}; anomalias de inventário: {len(problemas)}")
    print("Memória: somente uma gravação bruta é carregada de cada vez.")
    print("Para interromper: Ctrl+C; os NPZ já salvos permanecem no disco.")
    status = []
    for i, g in enumerate(gravacoes, 1):
        destino = CACHE_CENTRADO / (g.id + ".npz")
        if destino.exists() and not args.refazer:
            status.append({"id": g.id, "status": "cache", "janelas": "", "duracao_s": "", "motivo": ""})
            print(f"[{i}/{len(gravacoes)}] cache: {g.id}", flush=True)
            continue
        inicio = time.monotonic()
        print(f"[{i}/{len(gravacoes)}] processando: {g.id}", flush=True)
        try:
            emg = ler_csv_numerico(g.emg, 128)
            angulos = ler_csv_numerico(
                g.ang,
                17,
                permitir_nan_fora_intervalo=True
            )
            feat = extrair_caracteristicas(emg, angulos)
            temporario = CACHE_CENTRADO / (g.id + ".temporario.npz")
            np.savez_compressed(temporario, **feat, tarefa=np.int16(g.tarefa),
                                frequencia=np.float32(g.frequencia), repeticao=np.int16(g.repeticao))
            temporario.replace(destino)
            n = len(feat["tempo_s"])
            duracao = round(time.monotonic() - inicio, 2)
            status.append({"id": g.id, "status": "ok", "janelas": n,
                           "duracao_s": duracao, "motivo": ""})
            print(f"  OK: {n} janelas; {duracao}s; {destino.name}", flush=True)
            del emg, angulos, feat
        except Exception as e:
            status.append({"id": g.id, "status": "erro", "janelas": 0,
                           "duracao_s": "", "motivo": str(e)})
            print(f"  [ERRO] {type(e).__name__}: {e}", flush=True)
        finally:
            gc.collect()
            gravar_status(status)
    print("\nConcluído. Verifique", C.RESULTADOS / "01_processamento_centrado.csv")
    print("A revisão de qualidade da cinemática NÃO foi feita automaticamente.")


if __name__ == "__main__":
    main()
