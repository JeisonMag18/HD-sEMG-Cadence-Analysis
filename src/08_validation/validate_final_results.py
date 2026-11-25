"""Experimento de auditoria de R² e relação entre excursão angular e RMS.

Se a detecção automática não encontrar os CSVs corretos:
    python validate_final_results.py --rms "CAMINHO_DO_CSV_RMS" --ridge "CAMINHO_DO_CSV_RIDGE"

"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent
ROOT = BASE / "resultados_ia753"
OUT = ROOT / "52_validacao_final_artigo"


def norm(x):
    s = str(x).strip().lower()
    return re.sub(r"[^a-z0-9]+", "", s.replace("²", "2"))


def read_csv(p):
    for sep in (None, ",", ";", "\t"):
        try:
            d = pd.read_csv(p, sep=sep, engine="python", encoding="utf-8-sig")
            if d.shape[1] >= 2 and len(d):
                d.columns = [str(c).strip() for c in d.columns]
                return d
        except Exception:
            pass
    return None


def col(d, *names):
    mp = {norm(c): c for c in d.columns}
    for n in names:
        if norm(n) in mp:
            return mp[norm(n)]
    return None


def num(v):
    return pd.to_numeric(v.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce")


def rate(v):
    s = str(v).strip().lower().replace(",", ".")
    try:
        x = float(s)
        if abs(x - 0.5) < 0.02: return "0.50"
        if abs(x - 0.75) < 0.02: return "0.75"
    except ValueError:
        pass
    s = norm(s)
    if s in ("05", "050", "50", "05hz", "050hz", "50hz"): return "0.50"
    if s in ("075", "75", "075hz", "75hz"): return "0.75"
    return None


def task(v):
    s = str(v).strip().upper()
    m = re.search(r"(?:^|\D)(?:T\s*)?([1-8])(?:\D|$)", s)
    if m: return "T" + m.group(1)
    labels = {"INDEX":"T1", "MIDDLE":"T2", "RING":"T3", "THUMB":"T4"}
    return labels.get(s)


def region(v):
    s = str(v).upper()
    if "EDC" in s: return "EDC"
    if "FDS" in s: return "FDS"
    return None


def is_rms_label(v):
    s = norm(v)
    return ("rms" in s and "mnf" not in s and "mdf" not in s)


def is_rms_mnf(v):
    s = norm(v)
    return "rms" in s and "mnf" in s and "mdf" not in s and "smooth" not in s


def files():
    return [p for p in ROOT.rglob("*.csv") if OUT not in p.parents]


def ranking(p, kind):
    n = p.as_posix().lower()
    return (12 if f"/{kind}_" in n or f"\\{kind}_" in n else 0) + (8 if kind == "48" and ("ridge" in n or "ablation" in n) else 0) + (8 if kind == "37" and "rms" in n else 0) + (4 if "participant" in n or "subject" in n else 0)


def excursion(paths):
    matches = []
    for p in paths:
        if "51_" not in str(p): continue
        d = read_csv(p)
        if d is None: continue
        tc = col(d, "task", "tarefa")
        ac = col(d, "mean_delta_deg")
        a0 = col(d, "mean_050_deg")
        ah = col(d, "mean_075_deg")
        if tc and (ac or (a0 and ah)):
            q = pd.DataFrame({"task": d[tc].map(task)})
            q["delta_deg"] = num(d[ac]) if ac else num(d[ah]) - num(d[a0])
            if a0: q["pct_excursion"] = 100*q["delta_deg"] / num(d[a0])
            else: q["pct_excursion"] = np.nan
            ph = col(d,"p_holm_8", "p_holm")
            if ph: q["p_holm"] = num(d[ph])
            q = q.dropna(subset=["task", "delta_deg"])
            if len(q) == 8: matches.append((p, q))
    return matches[0] if matches else (None, None)


def task_rms_candidate(p, d):
    tc = col(d,"task","tarefa","task_id","movement","task_code")
    rc = col(d,"region","muscle","muscle_region","array","regiao")
    if not (tc and rc): return None
    if not ("rms" in p.name.lower() or any("rms" in norm(c) for c in d.columns)): return None
    tmp = pd.DataFrame({"task":d[tc].map(task),"region":d[rc].map(region)})
    pc = col(d,"delta_pct", "change_pct", "pct_change", "percent_change", "rms_delta_pct", "rms_change_pct", "delta_percent", "rms_pct", "change_percent", "pct", "percentage_change", "change_of_means_pct")
    if pc:
        tmp["rms_pct"] = num(d[pc])
    else:
        # Colunas largas para médias nos dois ritmos.
        cnorm = {norm(c):c for c in d.columns}
        c0 = next((v for k,v in cnorm.items() if ("050" in k or "05hz" in k) and ("rms" in k or "mean" in k)),None)
        c1 = next((v for k,v in cnorm.items() if ("075" in k or "075hz" in k) and ("rms" in k or "mean" in k)),None)
        if c0 and c1:
            x,y = num(d[c0]), num(d[c1])
            tmp["rms_pct"] = 100*(y-x)/x
        else:
            # Tabela longa: linhas por participante/cadência; calcular médias por participante e tarefa.
            cad = col(d,"cadence", "cadence_hz", "freq", "frequency", "frequency_hz", "cadencia", "cadencehz")
            val = col(d,"rms", "rms_mean", "mean_rms", "rms_uv", "rms_uV", "value")
            subj = col(d,"subject", "subject_id", "participant", "participant_id", "sub", "subjectid")
            if cad and val:
                tmp["cadence"] = d[cad].map(rate)
                tmp["value"] = num(d[val]); tmp["subject"] = d[subj].astype(str) if subj else "global"
                tmp = tmp.dropna(subset=["task","region","cadence","value"])
                temp = tmp.groupby(["subject","task","region","cadence"],as_index=False)["value"].mean()
                pv = temp.pivot_table(index=["subject","task","region"],columns="cadence",values="value").reset_index()
                if not all(c in pv for c in ["0.50","0.75"]): return None
                pv = pv.dropna(subset=["0.50","0.75"])
                means = pv.groupby(["task","region"],as_index=False)[["0.50","0.75"]].mean()
                means["rms_pct"] = 100*(means["0.75"]-means["0.50"])/means["0.50"]
                tmp = means
            else: return None
    tmp = tmp.dropna(subset=["task","region","rms_pct"])
    q = tmp.groupby(["task","region"],as_index=False)["rms_pct"].mean()
    if q["task"].nunique() < 6 or q["region"].nunique() < 2: return None
    return q


def find_rms(paths, explicit):
    cand = []
    for p in ([explicit] if explicit else sorted(paths, key=lambda x: ranking(x,"37"),reverse=True)):
        if not p or not p.exists(): continue
        d=read_csv(p)
        if d is None: continue
        q=task_rms_candidate(p,d)
        if q is not None:
            cand.append((p,q))
            if explicit: break
    return cand


def find_ridge(paths, explicit):
    result=[]
    for p in ([explicit] if explicit else sorted(paths,key=lambda x:ranking(x,"48"),reverse=True)):
        if not p or not p.exists(): continue
        filename = str(p).lower()
        if not explicit and not ("48_" in filename or "ridge" in filename or "ablation" in filename): continue
        d=read_csv(p)
        if d is None: continue
        fc=col(d,"feature_set", "features", "feature", "model", "descriptor", "input", "feature_combination", "feature_combo")
        cc=col(d,"cadence", "cadence_hz", "frequency", "frequency_hz", "cadencia", "freq", "hz")
        rc=col(d,"r2_vw", "r2vw", "r2_variance_weighted", "r2_variance_weighted_mean", "r2_mean", "r2", "r_squared", "r2_score", "mean_r2_vw")
        if not (fc and cc and rc): continue
        q=d[d[fc].map(is_rms_mnf)].copy()
        if q.empty: continue
        q["cadence52"]=q[cc].map(rate);q["r252"]=num(q[rc])
        q=q.dropna(subset=["cadence52","r252"])
        if q["cadence52"].nunique()<2: continue
        sc=col(q,"subject","subject_id","participant","participant_id","sub","subjectid")
        tmp=[]
        for cadence in ("0.50","0.75"):
            a=q[q.cadence52==cadence]
            if sc:
                a=a.groupby(sc,as_index=False)["r252"].mean()
            tmp.append({"cadence":cadence,"r2_mean":a.r252.mean(),"n_rows":len(a),"participant_column":str(sc or "NOT_FOUND"),"source":str(p)})
        result.append((p,pd.DataFrame(tmp),len(q),bool(sc)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rms",type=Path,help="CSV com variação de RMS por tarefa e região")
    parser.add_argument("--ridge",type=Path,help="CSV das métricas do experimento 48")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not ROOT.exists():
        print(f"ERRO: pasta não encontrada: {ROOT}");return
    paths=files()
    report=[f"Arquivos CSV encontrados: {len(paths)}"]
    exp_path,exc=excursion(paths)
    if exc is None:
        report.append("[PENDENTE] Não encontrei CSV resumo do experimento 51 com task e mean_delta_deg.")
    else:
        report.append(f"[OK] Excursão angular: {exp_path}")
        report.append(f"   tarefas: {len(exc)}; médias de delta: {exc['delta_deg'].round(4).tolist()}")
    rms_candidates=find_rms(paths,args.rms)
    if not rms_candidates:
        report.append("[PENDENTE] CSV de RMS por tarefa não identificado. Informe --rms CAMINHO.csv")
        report.append("   Possíveis arquivos: " + "; ".join(str(x) for x in paths if "37_" in str(x) or "rms" in x.name.lower())[:1400])
    else:
        report.append("Possíveis fontes RMS: " + "; ".join(str(p) for p,_ in rms_candidates[:5]))
    if exc is not None and rms_candidates:
        p,rms=rms_candidates[0]
        a=exc.merge(rms,on="task",how="inner").sort_values(["region","task"])
        rows=[]
        for reg in ["EDC","FDS"]:
            d=a[a.region==reg].dropna(subset=["delta_deg","rms_pct"])
            if len(d)>=4:
                rho,pvalue=spearmanr(d["delta_deg"],d["rms_pct"])
                rho_pct,pvalue_pct=spearmanr(d["pct_excursion"],d["rms_pct"])
                rows.append({"region":reg,"n_tasks":len(d),"spearman_deg":rho,"p_deg":pvalue,"spearman_pct":rho_pct,"p_pct":pvalue_pct})
                report.append(f"[RESULTADO] {reg}, n={len(d)}: Spearman delta graus vs RMS% = {rho:.4f} (p={pvalue:.4f}); excursão% vs RMS% = {rho_pct:.4f} (p={pvalue_pct:.4f})")
        if rows:
            a.to_csv(OUT/"52_task_excursion_rms.csv",index=False)
            pd.DataFrame(rows).to_csv(OUT/"52_spearman_resumo.csv",index=False)
            report.append(f"[OK] Tabelas salvas usando RMS de: {p}")
        else: report.append("[PENDENTE] RMS sem pelo menos quatro tarefas comparáveis em cada região")
    ridge_candidates=find_ridge(paths,args.ridge)
    if not ridge_candidates:
        report.append("[PENDENTE] CSV com R2 por cadência e RMS+MNF não localizado; informe --ridge CAMINHO.csv")
        report.append("   Possíveis arquivos: " + "; ".join(str(x) for x in paths if "48_" in str(x) or "ridge" in x.name.lower())[:1400])
    else:
        # Priorizar CSV de métricas por participante, depois análise 48.
        ridge_candidates.sort(key=lambda z:(z[3],ranking(z[0],"48")),reverse=True)
        p,summary,n,has_subject=ridge_candidates[0]
        summary.to_csv(OUT/"52_ridge_r2_audit.csv",index=False)
        report.append(f"[OK] R² RMS+MNF encontrado em: {p}")
        for _,r in summary.iterrows():
            reported=0.7711
            status="CONFERE" if abs(r.r2_mean-reported)<0.00005 else "DIVERGE"
            report.append(f"[RESULTADO] {r.cadence} Hz: R² médio={r.r2_mean:.6f}; artigo={reported:.4f}; {status}; n={int(r.n_rows)}")
        if not has_subject:
            report.append("[ATENÇÃO] CSV sem coluna de participante: média não é auditoria independente, confirmar CSV individual")
        elif any(summary.n_rows!=21):
            report.append("[ATENÇÃO] Esperados 21 participantes por cadência: conferir agrupamento")
    script=BASE/"valide_angular_excursion.py"
    if script.exists():
        lines=script.read_text(encoding="utf-8",errors="replace").splitlines()
        hits=[f"L{i}: {line.strip()[:150]}" for i,line in enumerate(lines,1) if any(s in line.lower() for s in ["percentile","quantile","groupby","reference", "finger", "4, 7, 10", "median", "repetition"])]
        (OUT/"52_trechos_metodo_experimento51.txt").write_text("\n".join(hits[:90]),encoding="utf-8")
        report.append("[OK] Trechos relevantes do script 51 salvos; leitura humana necessária para validar método")
    (OUT/"52_diagnostico.txt").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("\n".join(report))
    print(f"\nSaída: {OUT}")

if __name__ == "__main__":
    main()
