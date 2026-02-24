import os, subprocess
import time

list_cmds = [
    "python main.py -i 7 -d cdm",
    "python main.py -i 7 -d eti",
    "python main.py -i 7 -d trc"
]

for cmd in list_cmds:
    print(f"Đang bắt đầu: {cmd}")
    # shell=True cho phép chạy lệnh như một chuỗi CMD
    # check=True sẽ báo lỗi nếu lệnh train bị crash
    subprocess.run(cmd, shell=True, check=True) 
    print(f"Hoàn thành: {cmd}")