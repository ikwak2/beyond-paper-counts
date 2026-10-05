"""Refit archived models using public inputs; write only under outputs/model_refit."""
from pathlib import Path
import argparse,importlib.util,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap',type=int,default=2000,help='Use 2000 for the frozen manuscript specification')
    args=parser.parse_args()
    if args.bootstrap<20:parser.error('At least 20 resamples required')
    path=ROOT/'source/scripts/84_analyze_epj_distance_full.py'
    spec=importlib.util.spec_from_file_location('frozen_distance_models',path)
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    a.ROOT=ROOT;a.DATA=ROOT/'data/model_inputs/entrant_model_inputs.csv'
    a.OUT=ROOT/'outputs/model_refit';a.TABLES=a.OUT/'tables';a.FIGURES=a.OUT/'figures';a.MANIFESTS=a.OUT/'manifests';a.REPORT=a.OUT/'MODEL_REFIT_RESULTS.md'
    a.OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'source/config/epj_distance_based_analysis_v10.json').read_text());config['bootstrap_resamples']=args.bootstrap
    a.CONFIG=a.OUT/'refit_config.json';a.CONFIG.write_text(json.dumps(config,indent=2))
    a.PROTOCOL=ROOT/'source/docs/epj_distance_based_analysis_protocol_v1.0.md'
    a.IMPLEMENTATION=ROOT/'source/docs/epj_distance_model_implementation_note_v1.0a.md'
    original_read=pd.read_csv
    def read_with_internal_counter(file,*pos,**kw):
        d=original_read(file,*pos,**kw)
        if Path(file)==a.DATA:d['entrant_id']=np.arange(len(d))
        return d
    pd.read_csv=read_with_internal_counter
    try:a.main()
    finally:pd.read_csv=original_read
    check={}
    for generated in sorted(a.TABLES.glob('*.csv')):
        frozen=ROOT/'data/distance'/generated.name
        if not frozen.exists():continue
        actual=pd.read_csv(generated);expected=pd.read_csv(frozen)
        numeric=list(expected.select_dtypes(include='number').columns)
        for c in expected.columns:
            if c not in numeric:assert actual[c].fillna('').astype(str).equals(expected[c].fillna('').astype(str)),(generated.name,c)
        diff=np.abs(actual[numeric].to_numpy()-expected[numeric].to_numpy())
        same=np.allclose(actual[numeric],expected[numeric],rtol=1e-8,atol=1e-8,equal_nan=True)
        maxdiff=float(np.nanmax(diff)) if diff.size and np.any(np.isfinite(diff)) else 0.
        check[generated.name]={'matches_frozen':bool(same),'max_absolute_difference':maxdiff}
    report={'status':'PASS' if all(x['matches_frozen'] for x in check.values()) else 'MISMATCH','resamples':args.bootstrap,'checks':check,'comparison_tolerance':{'rtol':1e-8,'atol':1e-8},'scope':'Models refit from frozen distances and public covariates. No upstream collection or embedding recomputation.'}
    (a.OUT/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if args.bootstrap==2000:assert report['status']=='PASS','See outputs/model_refit/comparison.json'
if __name__=='__main__':main()
