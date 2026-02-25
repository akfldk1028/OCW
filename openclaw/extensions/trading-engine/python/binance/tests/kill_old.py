"""Kill all old Python 3.10.11 trading bot processes."""
import subprocess
import os

result = subprocess.run(
    ['tasklist', '/FI', 'IMAGENAME eq python3.exe'],
    capture_output=True, text=True
)
my_pid = os.getpid()
killed = 0
for line in result.stdout.strip().split('\n'):
    parts = line.split()
    if len(parts) > 1 and parts[1].isdigit():
        pid = int(parts[1])
        if pid == my_pid:
            continue
        ret = os.system(f'taskkill /PID {pid} /F')
        if ret == 0:
            killed += 1
            print(f'Killed PID {pid}')
        else:
            print(f'Failed to kill PID {pid}')
print(f'Total killed: {killed}')
