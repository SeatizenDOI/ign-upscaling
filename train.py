from argparse import Namespace, ArgumentParser

from src.ConfigParser import ConfigParser
from src.PathManager import PathManager
from src.UAVManager import UAVManager
from src.utils.lib_tools import print_header


def parse_args() -> Namespace:

    parser = ArgumentParser(prog="The point is the mask training", description="Workflow to WSSS training on UAV orthophoto using ASV prediction.")

    # Config.
    parser.add_argument("-cp", "--config_path", default="./configs/config_base.json", help="Path to the config file.")
    parser.add_argument("-ep", "--env_path", default="./.env", help="Path to the env file.")
    parser.add_argument("-oe", "--only_evaluation", action="store_true", help="Only perform evalution. Skip all the rest.")

    return parser.parse_args()


def main(opt: Namespace) -> None:

    print_header()
    
    # Initialize parser.
    cp = ConfigParser(opt)

    # Initialize path manager.
    pm = PathManager(cp.output_path)
    pm.setup(cp)

    # Download uav orthophoto
    uav_manager = UAVManager(cp, pm)



if __name__ == "__main__":
    opt = parse_args()
    main(opt)