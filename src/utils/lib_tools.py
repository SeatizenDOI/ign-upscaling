import enum
import torch
import pandas as pd
from pathlib import Path
from argparse import Namespace


def print_header():

    print("""
    
\t\t d888888b  d888b  d8b   db      db    db d8888b. .d8888.  .o88b.  .d8b.  db      d888888b d8b   db  d888b  
\t\t   `88'   88' Y8b 888o  88      88    88 88  `8D 88'  YP d8P  Y8 d8' `8b 88        `88'   888o  88 88' Y8b 
\t\t    88    88      88V8o 88      88    88 88oodD' `8bo.   8P      88ooo88 88         88    88V8o 88 88      
\t\t    88    88  ooo 88 V8o88      88    88 88~~~     `Y8b. 8b      88~~~88 88         88    88 V8o88 88  ooo 
\t\t   .88.   88. ~8~ 88  V888      88b  d88 88      db   8D Y8b  d8 88   88 88booo.   .88.   88  V888 88. ~8~ 
\t\t Y888888P  Y888P  VP   V8P      ~Y8888P' 88      `8888Y'  `Y88P' YP   YP Y88888P Y888888P VP   V8P  Y888P  
                                                                                                                                                                                                             
""")
    
def print_gpu_is_used() -> None:
    """ Print banner to show if gpu is used. """
    # Check if GPU available
    if torch.cuda.is_available():
        print("\n###################################################")
        print("Using GPU for training.")
        print("###################################################\n")
    else:
        print("GPU not available, using CPU instead.")


class Sources(enum.Enum):
    CSV_SESSION = 0
    FOLDER = 1
    SESSION = 2

def get_mode_from_opt(opt: Namespace) -> Sources | None:
    """ Retrieve mode from input option """
    mode = None

    if opt.enable_csv: 
        mode = Sources.CSV_SESSION
    elif opt.enable_folder: 
        mode = Sources.FOLDER
    elif opt.enable_session: 
        mode = Sources.SESSION

    return mode

def get_src_from_mode(mode: Sources, opt: Namespace) -> Path:
    """ Retrieve src path from mode """
    src = Path()

    if mode == Sources.CSV_SESSION:
        src = Path(opt.path_csv_file)
    elif mode == Sources.FOLDER:
        src = Path(opt.path_folder)
    elif mode == Sources.SESSION:
        src = Path(opt.path_session)

    return src


def get_list_rasters(opt: Namespace) -> list[Path]:
    """ Retrieve list of rasters from input """

    list_sessions: list[Path] = []

    mode = get_mode_from_opt(opt)
    if mode == None: return list_sessions

    src = get_src_from_mode(mode, opt)

    if mode == Sources.SESSION:
        list_sessions = [src]

    elif mode == Sources.FOLDER:
        list_sessions = sorted([s for s in src.iterdir() if s.is_file() and s.suffix.lower() == ".tif"])
    
    elif mode == Sources.CSV_SESSION:
        if src.exists():
            df_ses = pd.read_csv(src)
            list_sessions = [Path(row.root_folder, row.ortho_name) for row in df_ses.itertuples(index=False)]

    return list_sessions