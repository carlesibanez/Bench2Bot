"""Launch the UR5e Titration VLA Data Collection GUI."""

import sys
from pathlib import Path

# The Robotiq driver lives alongside the collector directory in Outreach.
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from vla_collector.gui import main


if __name__ == "__main__":
    main()
