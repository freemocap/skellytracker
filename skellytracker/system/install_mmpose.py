import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

def run_command(command: str):
    try:
        subprocess.run(command, shell=True, check=True)
        logger.info(f"Command `{command}` executed successfully")
    except subprocess.CalledProcessError:
        logger.error(f"Command `{command}` failed to execute")
        return False
    return True

def install_packages(package_commands: dict):
    for package, command in package_commands.items():
        success = run_command(command)
        if not success:
            raise RuntimeError(f"Failed to install {package}. Try manual installation.")

def install_mmpose():
    package_commands = {
        "mmengine": "mim install mmengine",
        "mmcv": "mim install \"mmcv>=2.0.1\"",
        "mmdet": "mim install \"mmdet>=3.1.0\"",
        "mmpose": "mim install \"mmpose>=1.1.0\""
    }

    install_packages(package_commands)
    # verify_mmpose_installation()

def verify_mmpose_installation():
    config_status = run_command("mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest .")
    if not config_status:
        raise RuntimeError("Failed to download config and checkpoint files.")

    run_inference_demo()
    check_output_exists()

def run_inference_demo():
    demo_command = " ".join([
        "python",
        "skellytracker/system/mmpose_image_demo.py",
        "skellytracker/system/mmpose_demo_image.png",
        "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
        "td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth",
        "--out-file mmpose_test_results.jpg",
        "--draw-heatmap"
    ])

    demo_status = run_command(demo_command)
    if not demo_status:
        raise RuntimeError("Inference demo failed to complete.")

def check_output_exists():
    if not Path("mmpose_test_results.jpg").is_file():
        raise RuntimeError("The output file from the inference demo does not exist.")