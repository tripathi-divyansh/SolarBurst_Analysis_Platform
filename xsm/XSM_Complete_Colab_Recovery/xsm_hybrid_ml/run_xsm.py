#!/usr/bin/env python3
"""Analyze standard XSM Level-2 LC files; writes review products, never labels."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xsm_burst import __version__
from xsm_burst.xsm import load_xsm_level2
from xsm_burst.pipeline import analyze
from xsm_burst.serialization import save_analysis


def review_plots(result, out):
    lc, background, table = result['lightcurve'], result['baseline'], result['catalog']
    folder = out/'review_plots'; folder.mkdir()
    def draw(ids, name, candidate=None):
        fig, ax=plt.subplots(figsize=(12,4), layout='constrained')
        ax.plot(lc.time[ids],lc.value[ids],'.-',ms=2,lw=.5,label='Observed rate')
        ax.fill_between(lc.time[ids],lc.value[ids]-lc.error[ids],lc.value[ids]+lc.error[ids],alpha=.15,label='Reported 1-sigma error')
        ax.plot(lc.time[ids],background[ids],color='orange',lw=2,label='Estimated background')
        if candidate is not None:
            ax.axvspan(candidate.observed_start_s,candidate.observed_end_s,color='red',alpha=.2,label='Unconfirmed candidate')
            fit=result['fits'][candidate.candidate_id]
            if fit.get('success'):
                ax.plot(fit['fit_time_s'],fit['predicted'],color='purple',ls='--',lw=1,label='FRED fit (check flags)')
            ax.set_xlim(lc.time[ids[0]],lc.time[ids[-1]])
        ax.set(title=name,xlabel='Seconds since saved MET time origin',ylabel='Count rate (counts/s)')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.savefig(folder/(name.split(':')[0]+'.png'),dpi=140);plt.close(fig)
    segments=lc.segments()
    for i,ids in enumerate(segments,1):draw(ids,f'interval_{i:02d}')
    for i,(_,row) in enumerate(table.iterrows(),1):
        seg=next(s for s in segments if lc.time[s[0]]<=row.observed_peak_s<=lc.time[s[-1]])
        ids=seg[(lc.time[seg]>=row.observed_start_s-120)&(lc.time[seg]<=row.observed_end_s+120)]
        draw(ids,f'candidate_{i:03d}: review required',row)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/july16',help='A .lc file or directory containing .lc and matching .gti files')
    parser.add_argument('--out',type=Path,required=True,help='A new output directory; existing reviews are protected')
    args=parser.parse_args()
    paths=[args.input] if args.input.is_file() else sorted(args.input.rglob('*.lc'))
    if not paths:raise SystemExit('No .lc files found. Re-extract the dataset ZIP; raw L1 files cannot be used here.')
    if args.out.exists():raise SystemExit('Output exists. Choose a NEW --out directory to preserve review work.')
    args.out.mkdir(parents=True)
    manifest=[]
    print(f'XSM engine {__version__}; candidate extraction only, no trained real-data classifier',flush=True)
    for i,path in enumerate(paths,1):
        print(f'[{i}/{len(paths)}] {path.name}',flush=True)
        lc=load_xsm_level2(path)
        result=analyze(lc,group_id=path.stem)
        # A hash suffix disambiguates source files with the same basename.
        out=args.out/(path.stem+'_'+lc.metadata['source_sha256'][:8])
        save_analysis(result,out)
        review=result['catalog'].copy()
        for name in ['label','split','morphology_label','review_notes']:review[name]=''
        review.to_csv(out/'review_candidates.csv',index=False)
        np.savez_compressed(out/'input_arrays.npz',time=lc.time,value=lc.value,error=lc.error,
                            bin_width=lc.bin_width,exposure=lc.exposure,quality=lc.quality)
        review_plots(result,out)
        manifest.append({'source_file':path.name,'directory':out.name,'candidates':len(review),
                         'processed_exposure_s':result['manifest']['processed_exposure_s'],
                         'fully_reviewed':False})
        print(f'  {len(review)} unconfirmed candidates; {result["manifest"]["processed_exposure_s"]:.2f} s processed',flush=True)
    (args.out/'run_summary.json').write_text(json.dumps(manifest,indent=2))
    print(f'Results: {args.out.resolve()}\nKeep labels blank unless independently reviewed. Download a backup before ending Colab.')

if __name__=='__main__':main()
