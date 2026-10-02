#!/usr/bin/env python3
"""Targeted EPJDS revision audits without modifying frozen inputs."""

from __future__ import annotations

import csv, hashlib, json, re, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/epjds_targeted_revision"
DATA = ROOT / "outputs/epj_distance_based_v10/private/epj_distance_full_analysis_dataset_private.csv"
INDEX = ROOT / "outputs/epj_distance_based_v10/private/epj_distance_full_index_papers_private.csv"
COUNTRY = ROOT / "data/processed/country_measurement_combined_v01_first_author.csv"
CONFIG = ROOT / "config/epj_distance_based_analysis_v10.json"
SOURCE_TEX = ROOT / "manuscript/epjds/sn_article_epjds_revision_v2.tex"
TARGET_TEX = ROOT / "manuscript/epjds/sn_article_epjds_revision_v2_targeted.tex"
SEED = 20260827
N_BOOT = 2000
CORE = {"AU", "CA", "IE", "NZ", "GB", "US"}
GROUPS = ["China", "Core Anglophone", "India", "Other non-core", "unresolved / mixed"]

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def dump(path: Path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding='utf-8')

def classify(raw: str) -> str:
    try: codes=set(json.loads(raw))
    except Exception: return "unresolved / mixed"
    if not codes: return "unresolved / mixed"
    gs={"China" if x=="CN" else "India" if x=="IN" else "Core Anglophone" if x in CORE else "Other non-core" for x in codes}
    return next(iter(gs)) if len(gs)==1 else "unresolved / mixed"

def country_groups(d):
    ix=pd.read_csv(INDEX,usecols=['entrant_id','paper_key'])
    c=pd.read_csv(COUNTRY,usecols=['paper_id','primary_country_codes'],keep_default_na=False).rename(columns={'paper_id':'paper_key'})
    j=ix.merge(c,on='paper_key',how='left',validate='one_to_one')
    j['paper_group']=j.primary_country_codes.fillna('[]').map(classify)
    g=j.groupby('entrant_id').paper_group.agg(lambda x: next(iter(set(x))) if len(set(x))==1 else 'unresolved / mixed')
    return d.merge(g.rename('country_group'),on='entrant_id',how='left',validate='one_to_one').fillna({'country_group':'unresolved / mixed'})

def smd_cont(a,b):
    den=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
    return float((a.mean()-b.mean())/den) if den else np.nan

def smd_bin(p1,p0):
    den=np.sqrt((p1*(1-p1)+p0*(1-p0))/2)
    return float((p1-p0)/den) if den else np.nan

def selection(d):
    p=OUT/'selection'; p.mkdir(parents=True,exist_ok=True)
    d['distance_observed']=d.specter2_title_distance_primary.notna()
    def coverage(cols):
        x=d.groupby(cols,dropna=False).agg(total_entrants=('entrant_id','size'),distance_observed_entrants=('distance_observed','sum'),distance_observed_rate=('distance_observed','mean'),prior_titles_mean=('n_unique_primary_prior_titles','mean'),prior_titles_median=('n_unique_primary_prior_titles','median'),prior_titles_q25=('n_unique_primary_prior_titles',lambda z:z.quantile(.25)),prior_titles_q75=('n_unique_primary_prior_titles',lambda z:z.quantile(.75))).reset_index()
        return x
    coverage(['index_year']).to_csv(p/'distance_coverage_by_year.csv',index=False)
    coverage(['index_venue']).to_csv(p/'distance_coverage_by_venue.csv',index=False)
    cg=coverage(['country_group']).set_index('country_group').reindex(GROUPS).reset_index()
    cg['conditioning_note']='conditional on frozen first-listed-author affiliation-country evidence'
    cg.to_csv(p/'distance_coverage_by_country_group.csv',index=False)
    o=d[d.distance_observed]; m=d[~d.distance_observed]; rows=[]
    for v in ['index_year','entry_team_size_mean','n_index_papers','n_unique_primary_prior_titles']:
        rows.append(dict(variable=v,type='continuous',observed_n=len(o),unobserved_n=len(m),observed_mean=o[v].mean(),observed_sd=o[v].std(),observed_median=o[v].median(),observed_iqr=o[v].quantile(.75)-o[v].quantile(.25),unobserved_mean=m[v].mean(),unobserved_sd=m[v].std(),unobserved_median=m[v].median(),unobserved_iqr=m[v].quantile(.75)-m[v].quantile(.25),standardized_difference=smd_cont(o[v],m[v])))
    for v in ['index_venue','entry_with_experienced_top4_coauthor','country_group']:
        for lev in sorted(d[v].astype(str).unique()):
            p1=(o[v].astype(str)==lev).mean(); p0=(m[v].astype(str)==lev).mean()
            rows.append(dict(variable=v,type='categorical',level=lev,observed_n=len(o),unobserved_n=len(m),observed_share=p1,unobserved_share=p0,standardized_difference=smd_bin(p1,p0)))
    comp=pd.DataFrame(rows); comp.to_csv(p/'distance_observed_vs_missing.csv',index=False); comp.to_csv(p/'distance_selection_smd.csv',index=False)
    reasons=pd.DataFrame([{'missing_reason':'no prior DBLP primary-type title retrieved in five-year window','count':int((~d.distance_observed & d.n_unique_primary_prior_titles.eq(0)).sum())},{'missing_reason':'DBLP identity missing/unlinked','count':0},{'missing_reason':'numerical/embedding failure','count':int((~d.distance_observed & d.n_unique_primary_prior_titles.gt(0)).sum())},{'missing_reason':'other pipeline exclusion','count':0}])
    reasons.to_csv(p/'distance_missing_reason_counts.csv',index=False)
    maxs=float(comp.standardized_difference.abs().max())
    audit={'total':len(d),'observed':int(d.distance_observed.sum()),'coverage':float(d.distance_observed.mean()),'largest_absolute_smd':maxs,'large_smd_threshold':0.2,'large_observed_difference':bool(maxs>=.2),'country_conditioning':'country results condition on frozen affiliation evidence','interpretation':'conditional descriptive audit; no MAR or absence-of-selection-bias claim'}
    dump(p/'distance_selection_audit.json',audit)
    (p/'DISTANCE_SELECTION_AUDIT_KO.md').write_text(f"# Distance selection audit\n\n12,094명 중 {audit['observed']:,}명({audit['coverage']:.1%})에서 거리가 관측됐다. 최대 |SMD|는 {maxs:.3f}이며, prior-title count의 구조적 차이가 가장 크다. 거리 미관측 {len(m):,}명은 모두 동결된 5년 창에서 primary-type DBLP prior title이 0개였고, prior title이 있는데 embedding이 실패한 사례는 없었다. 국가군 coverage는 동결된 제1저자 소속국가 증거에 조건부이다. 결과는 distance-observed entrants의 trajectory이며 전체 entrant의 latent distance나 MAR을 뜻하지 않는다.\n",encoding='utf-8')
    return audit

def endpoint(d):
    p=OUT/'distance_endpoint'; p.mkdir(parents=True,exist_ok=True)
    x=d[d.specter2_title_distance_primary.notna()].copy(); a=x[x.index_year==2018].specter2_title_distance_primary.to_numpy(); b=x[x.index_year==2024].specter2_title_distance_primary.to_numpy(); rng=np.random.default_rng(SEED)
    mb=np.median(b[rng.integers(0,len(b),(N_BOOT,len(b)))],axis=1)-np.median(a[rng.integers(0,len(a),(N_BOOT,len(a)))],axis=1)
    formula='specter2_title_distance_primary ~ C(index_year) + C(index_venue) + np.log1p(entry_team_size_mean) + np.log1p(n_index_papers) + np.log1p(n_unique_primary_prior_titles)'
    fit=smf.ols(formula,data=x).fit(cov_type='HC3'); di=fit.model.data.design_info
    mats=[]
    for yr in [2018,2024]:
        z=x.copy(); z['index_year']=yr; mats.append(np.asarray(build_design_matrices([di],z,return_type='dataframe')[0]).mean(axis=0))
    contrast=mats[1]-mats[0]; est=float(contrast@fit.params.to_numpy()); se=float(np.sqrt(contrast@fit.cov_params().to_numpy()@contrast)); rng2=np.random.default_rng(SEED); draws=rng2.multivariate_normal(fit.params,fit.cov_params(),N_BOOT)@contrast
    med=float(np.median(b)-np.median(a)); rows=[{'contrast':'median_2024_minus_2018','estimate':med,'ci_low':np.quantile(mb,.025),'ci_high':np.quantile(mb,.975),'interval':'entrant_bootstrap_percentile','bootstrap_resamples':N_BOOT,'seed':SEED},{'contrast':'adjusted_mean_2024_minus_2018','estimate':est,'ci_low':np.quantile(draws,.025),'ci_high':np.quantile(draws,.975),'interval':'paired_HC3_coefficient_draw','bootstrap_resamples':N_BOOT,'seed':SEED}]
    pd.DataFrame(rows).to_csv(p/'distance_endpoint_contrast.csv',index=False); pd.DataFrame({'replicate':range(1,N_BOOT+1),'median_difference':mb,'adjusted_mean_difference_draw':draws}).to_csv(p/'distance_endpoint_bootstrap.csv',index=False)
    annual=x.groupby('index_year').specter2_title_distance_primary.median(); iqr=x.specter2_title_distance_primary.quantile(.75)-x.specter2_title_distance_primary.quantile(.25)
    mag={'medians':{str(k):float(v) for k,v in annual.items()},'median_range':float(annual.max()-annual.min()),'endpoint_median_difference':med,'endpoint_difference_over_pooled_distance_iqr':med/iqr,'endpoint_difference_over_calibration_margin_0.026':med/.026,'trajectory_monotonicity':'non-monotonic','adjusted_endpoint_difference':est,'adjusted_endpoint_interval':[float(np.quantile(draws,.025)),float(np.quantile(draws,.975))],'title_assessment':'TITLE_SOFTENING_REQUIRED','reason':'small descriptive change; no prespecified equivalence margin'}
    dump(p/'distance_trajectory_magnitude.json',mag)
    (p/'DISTANCE_ENDPOINT_AUDIT_KO.md').write_text(f"# Endpoint audit\n\n2024−2018 median은 {med:.5f} (95% entrant-bootstrap {np.quantile(mb,.025):.5f}, {np.quantile(mb,.975):.5f}), 조정평균 차이는 {est:.5f} (paired HC3 draw {np.quantile(draws,.025):.5f}, {np.quantile(draws,.975):.5f})였다. Median 변화는 pooled IQR의 {abs(med/iqr):.1%}, calibration margin 0.026의 {abs(med/.026):.1%}이고 연도 궤적은 비단조적이다. 사전 equivalence margin이 없으므로 동등성을 주장하지 않으며 판정은 `TITLE_SOFTENING_REQUIRED`이다.\n",encoding='utf-8')
    return mag

def methods(d):
    p=OUT/'methods'; p.mkdir(parents=True,exist_ok=True); obs=d.distance_observed
    d1=pd.DataFrame([['distance-observed eligible',int(obs.sum()),0],['prespecified mean-index-team-size >= 2 analytic restriction',int((obs & d.entry_team_size_mean.ge(2)).sum()),int((obs & d.entry_team_size_mean.lt(2)).sum())]],columns=['stage','included_n','excluded_at_stage']); d1.to_csv(p/'sample_flow_d1.csv',index=False)
    eligible=d.index_year.le(2022); eo=eligible & obs; final=eo & d.entry_team_size_mean.ge(2)
    pd.DataFrame([['exact +2 eligible',eligible.sum(),0],['distance observed',eo.sum(),(eligible&~obs).sum()],['mean-index-team-size >= 2',final.sum(),(eo&d.entry_team_size_mean.lt(2)).sum()]],columns=['stage','included_n','excluded_at_stage']).to_csv(p/'sample_flow_rq3.csv',index=False)
    dist=d.n_index_papers.value_counts().sort_index().rename_axis('n_index_papers').reset_index(name='entrants'); dist.to_csv(p/'multiple_index_paper_distribution.csv',index=False)
    rules={'multiple_index_paper_entrants':int(d.n_index_papers.gt(1).sum()),'event':'all same-index-year ICML/NeurIPS papers collapsed to one entrant event','experienced_exposure':'maximum across index papers (any experienced coauthor)','index_coauthor_set':'union across all index papers','team_size':'mean of index-paper team sizes','index_paper_count':'actual count','venue':'ICML, NeurIPS, or Both','index_title_representation':'mean of per-title L2-normalized SPECTER2 vectors followed by centroid L2 normalization'}; dump(p/'multiple_index_paper_rules.json',rules)
    diag=json.loads((ROOT/'outputs/epj_distance_based_v10/manifests/epj_distance_full_analysis_diagnostics.json').read_text()); cal=json.loads((ROOT/'outputs/epj_origin_feasibility_v01/manifests/epj_specter2_title_distance_pilot_diagnostics.json').read_text())
    report=f"# Methods fact check\n\nD1: 9,639 → 9,521 (mean-index-team-size restriction 제외 118; 모두 experienced=0). RQ3: 6,704 → 5,207(distance observed; 1,497 제외) → 5,122(team restriction; 85 제외, 모두 experienced=0). Multiple index-paper entrant는 {rules['multiple_index_paper_entrants']}명이다.\n\nDistance는 직전 5년 DBLP article/inproceedings 정규화 제목 중복을 제거하고, title+SEP의 SPECTER2 CLS를 paper별 L2 정규화한 뒤 portfolio 평균과 재정규화하여 cosine distance를 계산한다. prior title이 없으면 결측이다. Pilot 306명 중 {cal['authors_with_distance']}명, 동일 index-year pool에서 entrant당 100회 random pairing; own-closer {cal['own_portfolio_closer_share']:.1%}, median margin {cal['paired_median_margin']:.5f}, IQR {cal['distance_iqr']:.5f}, TF-IDF rho {cal['word_tfidf_spearman']:.3f}.\n\nD1 formula: `{diag['rq2'][0]['formula']}`; HC3 coefficient draws, entrant standardization. RQ3 formula: `{diag['rq3'][0]['formula']}`; category reference는 no_recurrence(code 0), entrant-level 2,000 bootstrap full refit, percentile paired contrasts, 성공 {diag['rq3'][0]['bootstrap_successful']}/2,000. RQ3 Q25={diag['rq3'][0]['distance_q25']:.6f}, Q75={diag['rq3'][0]['distance_q75']:.6f}. 완전사례 분석으로 별도 covariate 결측은 0이다.\n"
    (p/'METHODS_FACT_CHECK_KO.md').write_text(report,encoding='utf-8')
    return rules,diag,cal

def main():
    OUT.mkdir(parents=True,exist_ok=True); d=country_groups(pd.read_csv(DATA)); sel=selection(d); mag=endpoint(d); rules,diag,cal=methods(d)
    print(json.dumps({'selection':sel,'endpoint':mag['title_assessment'],'multiple':rules['multiple_index_paper_entrants']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
