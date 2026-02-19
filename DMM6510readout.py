import pyvisa
import datetime
import time
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from typing import Iterable, List, Tuple, Union
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# ---------------- Global Variables ---------------- #
measurement_running = False
times, x_axis, voltages = [], [], []
start_time = time.time()

Step = Tuple[float, float]   # (I_target [A], T_ramp [s])
PORT = 8003
SOCKET_TIMEOUT = 5.0
SSA_BIT = 6    # Sequencer Step Active
TWI_BIT = 3    # Trigger Wait


# ---------------- Tkinter GUI ---------------- #
root = tk.Tk()
root.title("DMM6510 Live Measurement")

csv_filename_var = tk.StringVar(value=datetime.datetime.now().strftime("%m%d%Y_%H%M%S"))
treshold_var     = tk.StringVar(value="0.2")
ramprate_var     = tk.StringVar(value="20")
maxcurrent_var   = tk.StringVar(value="500")
IP_PSU_var       = tk.StringVar(value="169.254.249.195")
IP_DMM_var       = tk.StringVar(value="169.254.254.52")

# --- GUI Layout --- #
ttk.Label(root, text="CSV Filename:").grid(row=0, column=0)
filename_entry = ttk.Entry(root, textvariable=csv_filename_var, width=20)
filename_entry.grid(row=0, column=1)

ttk.Label(root, text="PSU IP:").grid(row=0, column=2)
ttk.Entry(root, textvariable=IP_PSU_var, width=20).grid(row=0, column=3)

ttk.Label(root, text="DMM IP:").grid(row=0, column=4)
ttk.Entry(root, textvariable=IP_DMM_var, width=20).grid(row=0, column=5)

ttk.Label(root, text="QD threshold [mV]:").grid(row=1, column=0)
ttk.Entry(root, textvariable=treshold_var, width=20).grid(row=1, column=1)

ttk.Label(root, text="Ramp rate [A/s]:").grid(row=1, column=2)
ttk.Entry(root, textvariable=ramprate_var, width=20).grid(row=1, column=3)

ttk.Label(root, text="Max current [A]:").grid(row=1, column=4)
ttk.Entry(root, textvariable=maxcurrent_var, width=20).grid(row=1, column=5)

start_button = ttk.Button(root, text="Start Measurement")
start_button.grid(row=2, column=0)

stop_button = ttk.Button(root, text="Stop", state=tk.DISABLED)
stop_button.grid(row=2, column=1)

exit_button = ttk.Button(root, text="Exit")
exit_button.grid(row=2, column=3)

def KeithleyDMM_start():
    IP = IP_DMM_var.get()
    write_script_to_Keithley(IP)

# ---------------- Keithley Script ---------------- #
def write_script_to_Keithley(IP, dt):
    rm = pyvisa.ResourceManager()
    
    try:
        inst = rm.open_resource(f"TCPIP0::{IP}::inst0::INSTR")
    except Exception as e:
        messagebox.showinfo("Error", f"Could not connect to Keithley:\n{e}")
        pass
        return e

    with open("DMM6510.tsp", "r") as f:
        script_content = f.read()

    script_content = script_content.replace("SAMPLE_RATE", dt )
    
    inst.write("abort")
    inst.write('script.delete("QD")')
    inst.write("loadscript QD")
    inst.write(script_content)
    inst.write("endscript")
    inst.write("QD.save()")
    inst.write("QD.run()")

    return inst

# ---------------- File Handling ---------------- #
def open_file():
    os.makedirs("data", exist_ok=True)
    
    while True:
        filename = f"data\\{filename_entry.get()}.csv"
        if not os.path.exists(filename):
            return open(filename, "w", newline="")
        
        filename_entry.delete(0, tk.END)
        filename_entry.insert(0, datetime.datetime.now().strftime("%m%d%Y_%H%M%S"))


# ---------------- Measurement Start ---------------- #
def start_measurement():
    global measurement_running, file, csv_writer, thread_readdata
    global times, x_axis, voltages
    measurement_running = True

    times.clear()
    x_axis.clear()
    voltages.clear()

    KeithleyDMM_start()
    time.sleep(1)

    file = open_file()
    csv_writer = csv.writer(file)
    csv_writer.writerow(["timestamp", "current(A)", "voltage(V)"])

    start_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)

    thread_readdata = threading.Thread(target=read_data, daemon=True)
    thread_readdata.start()
    start_graph()

# ---------------- Live Graph ---------------- #
def start_graph():
    global fig, ax, canvas, line
    fig, ax = plt.subplots()
    line, = ax.plot([], [], marker="o", linestyle="-")
    
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=3, column=0, columnspan=6)
    
    update_graph()


def update_graph():
    if x_axis:
        line.set_xdata(x_axis)
        line.set_ydata(voltages)
        ax.relim()
        ax.autoscale_view()
        ax.set_xlabel("Current (A)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("DMM6510 Live Measurement")
        canvas.draw()

    if measurement_running:
        root.after(100, update_graph)  # 5 Hz


# ---------------- Data Collection ---------------- #
def read_data(inst):
    try:
        data = inst.read().strip()
    except Exception as e:
        print("exception at data collection:", e)
        pass
    return data

def read_save_data(inst):
    i = 0
    qd = 0
    qd_time = 0
    error = 0
    global measurement_running
    while measurement_running:
        print("reading")
        try:
            data = inst.read().strip()
            print(data)
            if data == "QD":
                qd = 1
                qd_time = time.time()
                continue

            t_str, v_str = data.split(",")

            csv_writer.writerow([t_str, v_str])
            file.flush()

            i += 1
            if i % 1 == 0:
                x_axis.append(float(t_str))
                voltages.append(float(v_str))
                i = 0

            if qd and time.time() > qd_time + 1:
                measurement_running = False

        except Exception as e:
            error += 1
            if error > 5:
                print("exception at data collection:", e)
                stop_measurement()

        time.sleep(0.01)  # 100 Hz


# ---------------- Stop ---------------- #
def stop_measurement(inst):
    global measurement_running, thread_readdata
    measurement_running = False

    start_button.config(state=tk.ACTIVE)
    stop_button.config(state=tk.DISABLED)

    # PSU.abort_PSU()

    if thread_readdata.is_alive():
        thread_readdata.join()

    try: file.close()
    except: pass

    try: inst.close()
    except: pass

    messagebox.showinfo("Stopped", "Measurement finished. File saved.")

def stop_instrument(inst):
    inst.write("abort")
    inst.write("ABORt")
    inst.write("*CLS")
    inst.write("*RST")

def exit_app():
    root.destroy()


# ---------------- Bind Buttons ---------------- #
start_button.config(command=start_measurement)
stop_button.config(command=stop_measurement)
exit_button.config(command=exit_app)


# ---------------- GUI Start ---------------- #
# root.mainloop()
