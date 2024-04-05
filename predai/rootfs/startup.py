import os
print("Bootstrap predai")

os.system("ls -l /")
os.system("ls -l /config")

if not os.path.exists("/config/predai.py")
  print("Copy initial python files")
  os.system("cp -r *.py /config")
print("Startup")
os.system("python3 /config/predai.py")


