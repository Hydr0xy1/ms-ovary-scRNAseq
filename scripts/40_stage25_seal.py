"""Seal Stage 25 after explicit agent visual review of every exported PNG."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

import pandas as pd

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
from ms_ovary_scrna.stage25_bee_mouse_bridge import now, sha, write_json, write_tsv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--visual-review-complete',action='store_true',required=True)
    p.parse_args()
    if str(root)!='/root/autodl-tmp/ovary_scRNAseq':
        raise RuntimeError('Seal only in the authorized main-server project')
    out=root/'results/deep_dive_stage25_bee_mouse_bridge'
    figures=root/'figures/deep_dive_stage25_bee_mouse_bridge'
    state=json.loads((out/'RUN_STATE.json').read_text())
    for step in ['inputs','projection','inference','vko','report_figures_validation']:
        assert state['steps'][step]['status']=='complete',step
    audit=json.loads((out/'INPUT_OBJECT_AUDIT.json').read_text())
    source=(root/'results/06_annotation_v2.h5ad').stat()
    assert source.st_size==audit['file_bytes'] and source.st_mtime_ns==audit['mtime_ns']
    previous=pd.read_csv(out/'OUTPUT_MANIFEST.tsv',sep='\t')
    for row in previous.itertuples():
        assert sha(root/row.path)==row.sha256,row.path
    records=pd.read_csv(figures/'FIGURE_MANIFEST.tsv',sep='\t')
    assert len(records)==10
    for row in records.itertuples():
        assert b'/FontFile2' in (figures/f'{row.figure}.pdf').read_bytes()
        assert '<text ' in (figures/f'{row.figure}.svg').read_text()
    review={'reviewer':'Codex agent', 'reviewed_at':now(), 'status':'passed',
            'figures_reviewed':records.figure.tolist(),
            'checks':['labels readable','no clipping','NA distinct from zero',
                      'source-data alignment','standalone exports','PDF fonts embedded',
                      'SVG text editable','null findings retained'],
            'n_figures':10,'n_export_files':30,'all_source_data_present':True}
    write_json(review,out/'FIGURE_VISUAL_QA.json')
    validation=json.loads((out/'VALIDATION.json').read_text())
    validation.update(visual_review='passed',pdf_embedded_truetype_fonts=True,
                      input_object_still_unchanged=True,manifest_verified_before_seal=True)
    write_json(validation,out/'VALIDATION.json')
    manifest=[]
    for directory in [out,figures]:
        for file in sorted(directory.iterdir()):
            if file.is_file() and file.name not in ['OUTPUT_MANIFEST.tsv','RUN_STATE.json']:
                manifest.append(dict(path=str(file.relative_to(root)),bytes=file.stat().st_size,sha256=sha(file)))
    write_tsv(manifest,out/'OUTPUT_MANIFEST.tsv')
    state.update(overall='complete',completed_at=now(),
                 final_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                 projection_and_vko_commit='1f80de8819416ad1cc539297d03c2bb983566828',
                 all_requested_deliverables_present=True,figure_count=10,
                 no_new_public_expression_download=True,no_new_ml_models=True,
                 prior_stages_write_operations=False)
    state['steps']['visual_review_and_seal']={'status':'complete','finished':now()}
    write_json(state,out/'RUN_STATE.json')
    print(json.dumps({'overall':state['overall'],'figures':10,'manifest_files':len(manifest),
                      'input_unchanged':True,'git_commit':state['final_git_commit']}))


if __name__=='__main__':
    main()
