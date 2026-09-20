"""Create Stage 25 report and standalone Python figures on the main server."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ms_ovary_scrna.stage25_reporting import finalize

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    if str(root)!='/root/autodl-tmp/ovary_scRNAseq':
        raise RuntimeError('Use authorized main-server project')
    finalize(root)
