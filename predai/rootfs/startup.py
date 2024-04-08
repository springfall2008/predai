import os
print("Bootstrap predai")

print("Copy initial python files")
os.system("cp /*.py /config")
  
if not os.path.exists("/config/predai.yaml"):
  print("Copy template config file")
  os.system("cp /*.yaml /config")

print("Startup")
os.system("python3 /config/predai.py")


