import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

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


# Function to write the files used to perform L1d simulation
def writeL1d(nozzle, pL, TL, pR, TR):

    if simData["Nsteps"]>5:
        print("\nWARNING: empirical function to estimate L1d runtime hasn't been tested for Nsteps>5. Check L1d results after simulation!")

    # Empirical function to compute simulation time based on chosen number of steps
    runtime_L1d = 0
    for n in range(0, simData["Nsteps"]):
        runtime_L1d += (2* geom["tube"]["l_tub"]) / (0.96**(n/2) * np.sqrt(1.4*287*TL)) + 0.001

    D_tub = geom["tube"]["D_tub"]
    D_ada = geom["adapter"]["D_ada"]
    eqDiamInlet = np.sqrt( 4 * geom[nozzle]["b"] * geom[nozzle]["h"] / np.pi )
    eqDiamThroat = np.sqrt( 4 * geom[nozzle]["b"] * geom[nozzle]["t"] / np.pi )

    x_tub = -geom["tube"]["l_tub"]
    x_ada = geom["adapter"]["l_ada"]
    x_con = geom["adapter"]["l_ada"] + geom[nozzle]["l_con"]
    x_thr = geom["adapter"]["l_ada"] + geom[nozzle]["l_con"] + geom[nozzle]["l_thr"]
    x_div = geom["adapter"]["l_ada"] + geom[nozzle]["l_con"] + geom[nozzle]["l_thr"] + geom[nozzle]["l_div"]
    x_out = geom["adapter"]["l_ada"] + geom[nozzle]["l_con"] + geom[nozzle]["l_thr"] + geom[nozzle]["l_div"] + geom[nozzle]["l_out"]

    # writing L1d master python script
    with open("./L1d/tubeNozzle.py", "w") as file:
        file.write(f"""# config file for ludwieg tube simulation with L1d\n"""
                   f"""config.title = 'ludwieg_tube'\n"""
                   f"""my_gm = add_gas_model('ideal-air-gas-model.lua')\n\n"""
                   f"""# Definition of tube walls\n"""
                   f"""add_break_point({x_tub:.5f} , {D_tub:.5f} )\t# left closed wall\n"""
                   f"""add_break_point( {0.000:.5f} , {D_tub:.5f} )\t# adapter start and interface location\n"""
                   f"""add_break_point( {x_ada:.5f} , {D_ada:.5f} )\t# adapter end\n"""
                   f"""add_break_point( {(x_ada+0.015):.5f} , {eqDiamInlet:.5f} )\t# start of rectangular section (equivalent diameter)\n"""
                   f"""add_break_point( {x_con:.5f} , {eqDiamInlet:.5f} )\t# nozzle inlet, start of convergent (equivalent diameter)\n"""
                   f"""add_break_point( {x_thr:.5f} , {eqDiamThroat:.5f} )\t# throat section (equivalent diameter)\n"""
                   f"""add_break_point( {x_div:.5f} , {eqDiamInlet:.5f} )\t# end of divergent (equivalent diameter)\n"""
                   f"""add_break_point( {x_out:.5f} , {eqDiamInlet:.5f} )\t# outlet section (equivalent diameter)\n\n"""
                   f"""# Creation of gas-path\n"""
                   f"""left_wall = VelocityEnd(x0={x_tub:.5f}, vel=0.00000)\n"""
                   f"""driver_gas1 = GasSlug(p={pL}, vel=0.00000, T={TL}, gmodel_id=my_gm, ncells=480)\n"""
                   f"""internal = GasInterface(x0={(x_tub*0.4):.5f})\n"""
                   f"""driver_gas2 = GasSlug(p={pL}, vel=0.00000, T={TL}, gmodel_id=my_gm, ncells=1500)\n"""
                   f"""interface = GasInterface(x0=0.0000)\n"""
                   f"""driven_gas = GasSlug(p={pR}, vel=0.00000, T={TR}, gmodel_id=my_gm, ncells=200)\n"""
                   f"""right_wall = FreeEnd(x0={x_out:.5f})\n"""
                   f"""assemble_gas_path(left_wall, driver_gas1, internal, driver_gas2, interface, driven_gas, right_wall)\n\n"""
                   f"""# Time-stepping parameters settings\n"""
                   f"""config.dt_init = 1.0e-6\n"""
                   f"""config.max_time = {(runtime_L1d*1000):.3f}e-3\n"""
                   f"""config.max_step = 5e6\n"""
                   f"""add_dt_plot(0.0, 0.05e-3, 0.01e-3)\n"""
                   f"""#add_dt_plot(2e-3, 0.1e-3, 0.02e-3)\n"""
                   f"""#add_dt_plot(22e-3, 0.05e-3, 0.01e-3)\n"""
                   f"""#add_dt_plot(28e-3, 0.1e-3, 0.02e-3)\n"""
                   f"""#add_dt_plot(46e-3, 0.05e-3, 0.01e-3)\n"""
                   f"""#add_dt_plot(53e-3, 0.1e-3, 0.02e-3)\n"""
                   f"""add_history_loc({x_ada:.5f})\n"""
                   )

    # writing shell script to perform the L1d simulation
    with open("./L1d/runL1d.sh", "w") as file:
        file.write(f"""# shell script to run L1d simulation and plotting\n#\n#\n"""
                   f"""prep-gas ideal-air.inp ideal-air-gas-model.lua\n"""
                   f"""l1d4-prep --job=tubeNozzle.py\n"""
                   f"""l1d4 --run-simulation --job=tubeNozzle.py\n"""
                   f"""#l1d4 --xt-data --job=tubeNozzle.py --var-name=p #--log10 --tindx-end=9999\n"""
                   )

    # writing gas model file .inp
    with open("./L1d/ideal-air.inp", "w") as file:
        file.write(f"""model = "IdealGas"\n"""
                   f"""species = {{'air'}}\n"""
                   )

# Function to detect approximate constant-value (plateau) regions in pressure and temperature histories
def detectConstReg(t, y, smooth_window=200, derivative_threshold_factor=0.4, min_points=1500):
    """
    Detect approximate constant-value (plateau) regions in a signal.
    Returns list of (t_start, t_end, y_mean).
    """
    t = np.asarray(t)
    y = np.asarray(y)

    # Smooth signal to reduce noise
    y_smooth = uniform_filter1d(y, size=smooth_window)

    # Derivative
    dydt = np.gradient(y_smooth, t)

    # Threshold for near-constant region
    threshold = np.std(dydt) * derivative_threshold_factor

    mask = np.abs(dydt) < threshold

    segments = []
    start = None
    for i in range(len(t)):
        if mask[i] and start is None:
            start = i
        elif not mask[i] and start is not None:
            end = i - 1
            if end - start >= min_points:
                y_mean = np.mean(y[start:end])
                segments.append((t[start], t[end], y_mean))
            start = None
    if start is not None:
        end = len(t) - 1
        if end - start >= min_points:
            segments.append((t[start], t[end], np.mean(y[start:end])))
    
    return segments, y_smooth

# Function to modify lines (BCs) in SU2 config file
def modifyLine(filename, thingToChange, newValue):
    # Read the content of the file
    with open(filename, 'r+') as file:
        lines = file.readlines()

        # Look for the line containing THING_TO_CHANGE and modify it
        for i, line in enumerate(lines):
            if line.strip().lower().startswith(thingToChange.lower()):
                # Modify the part after the equal sign with the new output value
                lines[i] = f"{thingToChange}= {newValue}\n"
                break

        # Move the file pointer to the beginning
        file.seek(0)
        
        # Write the modified content back to the file
        file.writelines(lines)
        
        # Truncate the remaining content (if any)
        file.truncate()

    file.close

# Sutherland's viscosity law
def sutherland(T):
    mu = 1.716e-5 * ((T/273)**(3/2)) * (273+111)/(T+111)
    return mu



with open("simData.json") as f:
    simData = json.load(f)

with open("utils/geom.json") as f:
    geom = json.load(f)


#region - L1d simulation

if simData["runL1d"]:

    L1d_start = time.time()
    L1d_dir = f"./L1d/"

    if not os.path.exists(L1d_dir):
        os.makedirs(L1d_dir)

    writeL1d(simData["nozzle"], simData["initCond"]["pL"], simData["initCond"]["TL"], simData["initCond"]["pR"], simData["initCond"]["TR"])
    print("L1d simulation files correctly written to ./L1d/\n")

    print("Running L1d simulation ...")

    with open(L1d_dir+f"""log_L1d.log""", "w") as logFile:
            proc = subprocess.Popen(
                ["bash", "runL1d.sh"],
                cwd=L1d_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            for line in proc.stdout:
                #print(line, end="")
                logFile.write(line)

            proc.wait()
    
    L1d_time = time.time() - L1d_start
    print(f"L1d simulation ended in {int(L1d_time/60)} minutes!\n")
    print("-" * 70  +  "\n"  +  "-" * 70)

'''
else:
    if not os.path.exists(f"./L1d/tubeNozzle/history-loc-0000.data"):
        sys.exit(f"ERROR: L1d run is disabled and input hystory for SU2 not found.")
'''

print("\n")

#endregion


#region - Detect constant plateaux

cols_L1d = ['t','x','vel','L_bar','rho','p','T','u','a','shear_stress','heat_flux','massf_air']

if simData["compPlateau"]:

    # load L1d simulation results
    resultsL1d = pd.read_csv("./L1d/tubeNozzle/history-loc-0000.data", sep=r"\s+", names=cols_L1d, comment='#')

    # pressure and temperature plots
    fig, ax = plt.subplots(); ax.grid()
    ax.plot(1000*resultsL1d["t"].values, resultsL1d["p"].values, "-", color="tab:blue")
    ax.set_ylabel("p [Pa]"); ax.set_xlabel("t [ms]")
    ax.set_title(f"""L: {int(simData["initCond"]["pL"]/1000)}kPa-{simData["initCond"]["TL"]:.2f}K , R: {int(simData["initCond"]["pR"]/1000)}kPa-{simData["initCond"]["TR"]:.2f}K""")
    fig.savefig("./L1d/inletPressure.pdf", format='png', dpi=1200)

    fig, ax = plt.subplots(); ax.grid()
    ax.plot(1000*resultsL1d["t"].values, resultsL1d["T"].values, "-", color="tab:blue")
    ax.set_ylabel("T [K]"); ax.set_xlabel("t [ms]")
    ax.set_title(f"""L: {int(simData["initCond"]["pL"]/1000)}kPa-{simData["initCond"]["TL"]:.2f}K , R: {int(simData["initCond"]["pR"]/1000)}kPa-{simData["initCond"]["TR"]:.2f}K""")
    fig.savefig("./L1d/inletTemperature.pdf", format='png', dpi=1200)

    # compute constant flow regions
    print(f"Computing steady flow regions from L1d results:\n")

    with open("./L1d/resultsL1d.csv", "w") as fileCSV:
        fileCSV.write(f"#Constant regions in L1d results, used as input for SU2 nozzle simulations\n#step_id;t_start[ms];t_end[ms];dt[ms];pressure[Pa];temperature[K]\n")
    fileCSV.close

    stepsL1d = pd.DataFrame(columns=["step_id", "t_start", "t_end", "dt", "p", "T"])
    presL1d, p_smooth = detectConstReg(resultsL1d['t'].values, resultsL1d['p'].values, 40, 0.2, 400)
    tempL1d, T_smooth = detectConstReg(resultsL1d['t'].values, resultsL1d['T'].values, 40, 0.3, 400)

    i = 0

    for (t1_p, t2_p, p_mean), (t1_T, t2_T, T_mean) in zip(presL1d, tempL1d):

        stepsL1d.loc[i] = [i, np.maximum(t1_p, t1_T), np.minimum(t2_p, t2_T), np.minimum(t2_p, t2_T)-np.maximum(t1_p, t1_T), p_mean, T_mean]
        print(f"""step{str(i)}: t1={(stepsL1d["t_start"].iloc[i]*1000):.2f}ms\tto t2={(stepsL1d["t_end"].iloc[i]*1000):.2f}ms,\twith p={stepsL1d["p"].iloc[i]:.2f}Pa and T={stepsL1d["T"].iloc[i]:.2f}K,  dt={(stepsL1d["dt"].iloc[i]*1000):.2f}ms""")

        with open("./L1d/resultsL1d.csv", "a") as fileCSV:
            fileCSV.write(f"""{str(i)};{stepsL1d["t_start"].iloc[i]:.6f};{stepsL1d["t_end"].iloc[i]:.6f};{stepsL1d["dt"].iloc[i]:.6f};{stepsL1d["p"].iloc[i]:.6f};{stepsL1d["T"].iloc[i]:.6f}\n""")
        fileCSV.close

        i+=1

    print("\nL1d steps correctly written to file.")

    if int(i) != int(simData["Nsteps"]):
        print("WARNING: input Nsteps is different from actual number of found steps! Check resultsL1d.csv file and plots!")

    print("-" * 70  +  "\n"  +  "-" * 70)

else:
    # load L1d steps from file
    stepsL1d = pd.read_csv("./L1d/resultsL1d.csv", delimiter=';', skiprows=2, names=["step_id", "t_start", "t_end", "dt", "p", "T"])

#endregion


#region - SU2 simulations

cols_SU2 = ["Inner_Iter",
        "rms[Rho]","rms[RhoU]","rms[RhoV]","rms[RhoW]","rms[RhoE]","rms[k]","rms[w]",
        "Avg_Massflow","Avg_Mach","Avg_Temp","Avg_Press","Avg_TotalTemp","Avg_TotalPress",
        "Avg_Massflow(OUTLET)","Avg_Massflow(THROAT)",
        "Avg_Mach(OUTLET)","Avg_Mach(THROAT)",
        "Avg_Temp(OUTLET)","Avg_Temp(THROAT)",
        "Avg_Press(OUTLET)","Avg_Press(THROAT)",
        "Avg_TotalTemp(OUTLET)","Avg_TotalTemp(THROAT)",
        "Avg_TotalPress(OUTLET)","Avg_TotalPress(THROAT)"]

cols_SU2extended = ["step_id"] + cols_SU2 + ["Avg_Rex(OUTLET)", "Avg_htot(OUTLET)"]
outDataSU2 = pd.DataFrame(columns=cols_SU2extended)
reportSU2 = []

if simData["runSU2"]:
    
    print(f"\nStarting SU2 nozzle simulation sequence...\n")
    su2_start = time.time()

    # load L1d steps from file
    #stepsL1d = pd.read_csv("./L1d/resultsL1d.csv", delimiter=';', skiprows=2, names=["step", "dt", "p", "T"])

    p_ref = stepsL1d.iloc[0]["p"]
    T_ref = stepsL1d.iloc[0]["T"]

    for step, row in stepsL1d.iterrows():

        workDir = f"./step{str(step)}/"
        cfgFile = f"step{str(step)}.cfg"
        cfgPath = workDir + cfgFile

        pR = simData["initCond"]["pR"]
        TR = simData["initCond"]["TR"]

        p_in = row["p"]
        T_in = row["T"]
        dt   = row["dt"]
        
        # check for overexpanded flow, if yes interrupt
        if step != 0:
            pExit_is = p_in * (1 + (simData["gamma"]-1)/2 * lastAvgMach**2) ** (-simData["gamma"]/(simData["gamma"]-1))

            if pExit_is < simData["initCond"]["pR"]:
                print(f"""WARNING: for step {step}, nozzle flow results to be overexpanded! Stopping SU2 simulations.""")
                break

        print(f"Running step {step}, logging to file...")
        step_start = time.time()

        # create folders for the steps RANS simulations
        if not os.path.exists(f"./step{str(step)}"):
            os.makedirs(f"./step{str(step)}")

        shutil.copy(f"./utils/steadyRANS_nozzle.cfg", cfgPath)

        # edit cfg file
        if step == 0:
            modifyLine(cfgPath, "RESTART_SOL", f"NO")
        else:
            modifyLine(cfgPath, "RESTART_SOL", f"YES")
            shutil.copy(f"./step{str(step-1)}/restart_flow.dat", workDir+"restart_flow.dat")

        modifyLine(cfgPath, "MARKER_RIEMANN",               f"""( INLET, TOTAL_CONDITIONS_PT, {p_in:.4f}, {T_in:.5f}, 1.0, 0.0, 0.0, OUTLET, STATIC_PRESSURE, {pR:.4f}, 0.0, 0.0, 0.0, 0.0)""")
        modifyLine(cfgPath, "MESH_FILENAME",                f"""../utils/mesh/{simData["mesh"]}.su2""")
        modifyLine(cfgPath, "ITER",                         f"""5000""")
        modifyLine(cfgPath, "FREESTREAM_PRESSURE",          f"""{p_ref:.4f}""")
        modifyLine(cfgPath, "FREESTREAM_TEMPERATURE",       f"""{T_ref:.6f}""")


        # running the current step simulation and saving log to file

        with open(workDir+f"log_step{str(step)}.log", "w") as logFile:
            proc = subprocess.Popen(
                ["SU2_CFD", str(cfgFile)],
                cwd=workDir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            for line in proc.stdout:
                #print(line, end="")
                logFile.write(line)

            proc.wait()

        time.sleep(10)

        # retrieving Avg_Mach(OUTLET) to estimate if next step is choked
        histFile = workDir + f"""/history*.csv"""
        files = glob.glob(histFile)
        if not files:
            raise FileNotFoundError(f"No CSV files found for pattern: {histFile}")

        # read the first matched file
        curHistory = pd.read_csv(files[0], delimiter=',', comment='"', names=cols_SU2)

        lastAvgMach = curHistory["Avg_Mach(OUTLET)"].iloc[-1]
        lastRow = curHistory.iloc[[-1]].copy()
        lastRow.insert(0, "step_id", step)

        # compute Rex
        mu = sutherland(lastRow["Avg_Temp(OUTLET)"].iloc[0])
        rho = lastRow["Avg_Press(OUTLET)"].iloc[0] / (simData["R"] * lastRow["Avg_Temp(OUTLET)"].iloc[0])
        vel = lastRow["Avg_Mach(OUTLET)"].iloc[0] * ( simData["gamma"] * simData["R"] * lastRow["Avg_Temp(OUTLET)"].iloc[0] )**0.5

        lastRow["Avg_Rex(OUTLET)"] = rho * vel / mu
        lastRow["Avg_htot(OUTLET)"] = lastRow["Avg_TotalTemp(OUTLET)"].iloc[0] * simData["gamma"] * simData["R"] / (simData["gamma"] - 1)

        # append to SU2 out dataframe
        if outDataSU2.empty:
            outDataSU2 = lastRow.copy()
        else:
            outDataSU2 = pd.concat([outDataSU2, lastRow], ignore_index=True)

        # check if file exists (to avoid rewriting header) and write all to file
        write_header = not os.path.exists("./results_SU2.csv")

        lastRow.to_csv(
            "./results_SU2.csv",
            mode="a",
            header=write_header,
            index=False
        )

        step_time = time.time() - step_start

        # report commands
        reportSU2.append({
            "case": f"step{str(step)}.cfg",
            "status": "success" if proc.returncode == 0 else "failed",
            "runtime_sec": round(step_time),
            "log_file": f"log_step{str(step)}.log"
        })

        if proc.returncode == 0:
            print(f"SU2 step{str(step)} completed in {int(step_time/60)} minutes.\n")
        else:
            print(f"SU2 step{str(step)} failed. Check log.")
            break

    su2_total = time.time() - su2_start

    # print to screen the simulation summary
    print("\nSimulation Summary")
    print("-" * 40)
    for r in reportSU2:
        print(f"{r['case']:10s} | {r['status']:8s} | {int(r['runtime_sec']/60)} min | Log: {r['log_file']}")
    print("-" * 40)
    print(f"Total SU2 runtime: {int(su2_total/60)} minutes")

    # save to file the simulation summary
    with open("SU2_summary.txt", "w") as f:
        f.write("\nSimulation Summary\n")
        f.write("-" * 40 + "\n")
        for r in reportSU2:
            f.write(
                f"{r['case']:10s} | {r['status']:8s} | "
                f"{int(r['runtime_sec']/60)} min | Log: {r['log_file']}\n"
            )
        f.write("-" * 40 + "\n")
        f.write(f"Total SU2 runtime: {int(su2_total/60)} minutes\n")


else:
    outDataSU2 = pd.read_csv("./results_SU2.csv", delimiter=',', skiprows=1, names=cols_SU2extended)


# writing the final resume file with input, intermediate and output
cols_out = [
    "pL", "TL", "pR", "TR",
    "p0", "T0", "p1", "T1", "p2", "T2",
    "Rex0", "htot0", "Rex1", "htot1", "Rex2", "htot2"
]

int_vals = []
out_vals = []

for i in range(3):
    if i < len(stepsL1d):
        out_vals.extend([stepsL1d.iloc[i]["p"], stepsL1d.iloc[i]["T"]])
        int_vals.extend([outDataSU2.iloc[i]["Avg_Rex(OUTLET)"], outDataSU2.iloc[i]["Avg_htot(OUTLET)"]])
    else:
        out_vals.extend([np.nan, np.nan])
        int_vals.extend([np.nan, np.nan])

final_vector = [
    simData["initCond"]["pL"], simData["initCond"]["TL"], simData["initCond"]["pR"], simData["initCond"]["TR"],
    *int_vals,
    *out_vals
]

df_final = pd.DataFrame([final_vector], columns=cols_out)
df_final.to_csv("GP_data.csv", index=False)

#endregion


'''
#region Plots

fig, ax = plt.subplots(); ax.grid()
ax.plot(1000*outDataSU2["Cur_Time"].to_numpy(), outDataSU2["Avg_Press(OUTLET)"].to_numpy(), '-', color='tab:orange')
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
ax.set_ylabel('p [Pa]')
ax.set_xlabel('t [ms]')
ax.set_title(f"""L: {int(simData["initCond"]["pL"]/1000)}kPa-{simData["initCond"]["TL"]:.2f}K , R: {int(simData["initCond"]["pR"]/1000)}kPa-{simData["initCond"]["TR"]:.2f}K""")
#ax.set_xlim([-0.5, 30])
ax.set_ylim([-0.5, 50000])
#plt.legend(["L1d", "su2"], loc="upper right")
#plt.show()
fig.savefig('./pressOUT.pdf', format='png', dpi=1200)

fig, ax = plt.subplots(); ax.grid()
ax.plot(1000*outDataSU2["Cur_Time"].to_numpy(), outDataSU2["Avg_TotalPress(OUTLET)"].to_numpy(), '-', color='tab:orange')
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
ax.set_ylabel('p [Pa]')
ax.set_xlabel('t [ms]')
ax.set_title(f"""L: {int(simData["initCond"]["pL"]/1000)}kPa-{simData["initCond"]["TL"]:.2f}K , R: {int(simData["initCond"]["pR"]/1000)}kPa-{simData["initCond"]["TR"]:.2f}K""")
#ax.set_xlim([-0.5, 30])
#ax.set_ylim([-0.5, 1000])
#plt.legend(["L1d", "su2"], loc="upper right")
#plt.show()
fig.savefig('./totpressOUT.pdf', format='png', dpi=1200)

fig, ax = plt.subplots(); ax.grid()
ax.plot(1000*outDataSU2["Cur_Time"].to_numpy(), outDataSU2["Avg_Mach(OUTLET)"].to_numpy(), '-', color='tab:orange')
ax.plot(1000*outDataSU2["Cur_Time"].to_numpy(), outDataSU2["Avg_Mach(THROAT)"].to_numpy(), '-', color='tab:blue')
ax.set_ylabel('M [-]')
ax.set_xlabel('t [ms]')
ax.set_title(f"""L: {int(simData["initCond"]["pL"]/1000)}kPa-{simData["initCond"]["TL"]:.2f}K , R: {int(simData["initCond"]["pR"]/1000)}kPa-{simData["initCond"]["TR"]:.2f}K""")
#ax.set_xlim([-0.5, 30])
ax.set_ylim([-0.5, 5])
plt.legend(["OUT", "THR"], loc="upper right")
#plt.show()
fig.savefig('./mach.pdf', format='png', dpi=1200)


fig, ax = plt.subplots(); ax.grid()
ax.plot(1000*outDataSU2["Cur_Time"].to_numpy(), outDataSU2["Avg_Rex(OUTLET)"].to_numpy(), '-', color='tab:orange')
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
ax.set_ylabel('Rex [-]')
ax.set_xlabel('t [ms]')
ax.set_title(f"""L: {int(simData["initCond"]["pL"]/1000)}kPa-{simData["initCond"]["TL"]:.2f}K , R: {int(simData["initCond"]["pR"]/1000)}kPa-{simData["initCond"]["TR"]:.2f}K""")
#ax.set_xlim([-0.5, 30])
#ax.set_ylim([-0.5, 1e8])
plt.legend(["OUT", "THR"], loc="upper right")
#plt.show()
fig.savefig('./reynolds.pdf', format='png', dpi=1200)



#endregion


'''
