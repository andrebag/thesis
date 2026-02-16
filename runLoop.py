import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import json
import glob
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from pathlib import Path
from datetime import datetime
from scipy.ndimage import uniform_filter1d
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# input
p_vec = [
  1079054.27, 1188467.23,  703723.58,  549276.66,  436032.5 ,  899718.95,
  1013621.46,  855744.41,  615053.48,  733783.79
  ] # Pa
 
 
T_vec = [
  416.8 , 357.01, 434.22, 393.89, 342.1 , 385.08, 306.32, 326.35, 302.74,
  363.92
  ]    # K


pR = 5e3
TR = 288.15


# cartella che contiene il caso base
BASE_DIR = Path("base")

# dove creare tutte le simulazioni
RUNS_DIR = Path(".")

# nome eseguibile python
PYTHON = "python3"

TRAIN_FILE = RUNS_DIR / "trainData.csv"


# functions

def make_case_name(p, T):

    # simulation directory name
    return f"p{int(p)}Pa_T{int(T)}K"



def update_simdata_json(json_path, pL, TL, pR, TR):

    with open(json_path) as f:
        simData = json.load(f)

    simData["initCond"]["pL"] = float(pL)
    simData["initCond"]["TL"] = float(TL)
    simData["initCond"]["pR"] = float(pR)
    simData["initCond"]["TR"] = float(TR)
    #simData["runL1d"] = False

    with open(json_path, "w") as f:
        json.dump(simData, f, indent=2)



def run_case(case_dir):

    # launch simulation

    log_file = case_dir / "digitalTwin.log"

    with open(log_file, "w") as log:

        proc = subprocess.Popen(

            [PYTHON, "digitalTwin.py"],

            cwd=case_dir,

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            universal_newlines=True,

            bufsize=1

        )

        for line in proc.stdout:

            log.write(line)

        proc.wait()

    return proc.returncode



def append_GP_to_train(case_dir, train_file):

    """
    Legge GP_data.csv dal case e lo appende a trainData.csv mantenendo stesso formato.
    """

    gp_file = case_dir / "GP_data.csv"

    if not gp_file.exists():

        raise FileNotFoundError(f"{gp_file} non trovato")


    df = pd.read_csv(gp_file)


    if len(df) != 1:

        raise RuntimeError(
            f"{gp_file} contiene {len(df)} righe (attesa 1)"
        )


    write_header = not train_file.exists()


    df.to_csv(
        train_file,
        mode="a",
        header=write_header,
        index=False
    )



# main

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def worker(args):
    p, T = args

    case_name = make_case_name(p, T)
    case_dir = RUNS_DIR / case_name

    print(f"\n=== Running case {case_name} ===")

    #if case_dir.exists():
    #    raise RuntimeError(f"Folder {case_dir} already exists")

    shutil.copytree(BASE_DIR, case_dir)
    #shutil.copy(BASE_DIR / "utils/steadyRANS_nozzle.cfg" , case_dir / "utils/steadyRANS_nozzle.cfg" )
    #shutil.copy(BASE_DIR / "utils/mesh/longNozzleM3.su2" , case_dir / "utils/mesh/longNozzleM3.su2" )
    #shutil.copy(BASE_DIR / "digitalTwin.py" , case_dir / "digitalTwin.py" )
    #shutil.copy(BASE_DIR / "simData.json" , case_dir / "simData.json" )
        
    json_path = case_dir / "simData.json"
    update_simdata_json(json_path, p, T, pR, TR)

    ret = run_case(case_dir)

    #if ret == 0:
    #    append_GP_to_train(case_dir, TRAIN_FILE)

    return case_name, ret


def main():

    if len(p_vec) != len(T_vec):
        raise ValueError("p_vec e T_vec must have same length")

    tasks = list(zip(p_vec, T_vec))

    # numero massimo di processi = core allocati
    # puoi anche fissarlo a mano: max_workers=10
    max_workers = int(os.environ.get("NSLOTS", os.cpu_count()))

    print(f"\nUsing {max_workers} parallel workers\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        futures = [executor.submit(worker, t) for t in tasks]

        for fut in as_completed(futures):

            case_name, ret = fut.result()

            if ret == 0:
                print(f"{case_name} completed.")
            else:
                print(f"{case_name} FAILED.")



if __name__ == "__main__":

    main()



dfs = []

for d in RUNS_DIR.glob("p*Pa_T*K"):
    f = d / "GP_data.csv"
    if f.exists():
        dfs.append(pd.read_csv(f))

pd.concat(dfs).to_csv(TRAIN_FILE, index=False)
