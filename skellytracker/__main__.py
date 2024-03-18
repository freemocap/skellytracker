# __main__.py
import sys
from pathlib import Path

base_package_path = Path(__file__).parent.parent
print(f"adding base_package_path: {base_package_path} : to sys.path")
sys.path.insert(0, str(base_package_path))  # add parent directory to sys.path

import logging

logger = logging.getLogger(__name__)

from skellytracker.RUN_ME import main


def cli_main():
    logger.info(f"Running as a script")
    if len(sys.argv) > 1:
        demo_tracker = str(sys.argv[1])
    else:
        demo_tracker = "mediapipe_holistic_tracker"
    main(demo_tracker=demo_tracker)

if __name__ == "__main__":
    cli_main()