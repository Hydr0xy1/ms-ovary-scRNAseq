"""Run independent Stage 25 on the authorized main server."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ms_ovary_scrna.stage25_bee_mouse_bridge import main

if __name__ == '__main__':
    main()
