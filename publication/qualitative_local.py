"""Recreate six qualitative panels using authorized, local private study data.

Outputs include clinical image pixels and identifiable provenance files. Keep the
entire output directory private; it must be outside this repository.
"""
from pathlib import Path
import argparse,os
if __package__:
    from .replay_figures import module,HERE
else:
    from replay_figures import module,HERE

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study-root',type=Path,required=True);p.add_argument('--out','--output',dest='out',type=Path,required=True)
    p.add_argument('--font-regular',type=Path,required=True);p.add_argument('--font-bold',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();repo=HERE.parent.resolve()
    if out==repo or repo in out.parents:p.error('Private output must be outside the GitHub repository.')
    for key,path in [('PUBLICATION_ARIAL_REGULAR',a.font_regular),('PUBLICATION_ARIAL_BOLD',a.font_bold)]:
        if not path.is_file():p.error('Font file not found.')
        os.environ[key]=str(path.resolve())
    out.mkdir(parents=True,exist_ok=True);m=module('qualitative')
    m.ROOT=a.study_root.resolve();m.OUT=out;m.RUN=m.ROOT/'strict_roi_results_4model_v1_2_0';m.DATA=m.ROOT/'çalışma_ds'
    m.main()
    print('Generated six clinical illustration pairs and private provenance. Do not publish this output directory as public data.')

if __name__=='__main__':main()
