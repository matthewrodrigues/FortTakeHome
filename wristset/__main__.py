"""Package entry point so ``python -m wristset`` runs the Phase-5 demo."""

import sys

from wristset.demo import main

if __name__ == "__main__":
    sys.exit(main())
