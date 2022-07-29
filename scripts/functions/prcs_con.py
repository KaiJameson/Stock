import time
import psutil

def pause_running_training():
    s = time.time()
    processes = {p.pid: p.info for p in psutil.process_iter(["name"])}
    python_processes_pids = []
    pause_list = []

    for process in processes:
        if processes[process]["name"] == "python.exe":
            python_processes_pids.append(process)

    for pid in python_processes_pids:
        if any("batch" in string for string in psutil.Process(pid).cmdline()):
            pause_list.append(pid)
        elif any("sym" in string for string in psutil.Process(pid).cmdline()):
            pause_list.append(pid)

    for pid in pause_list:
        psutil.Process(pid).suspend()

    print(f"Pausing python files took {time.time() - s}")
    return pause_list

def resume_running_training(pause_list):
    print(pause_list)
    for pid in pause_list:
        psutil.Process(pid).resume()



