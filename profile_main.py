import os
import time
from PIL import Image

os.system("python -m cProfile -o main_profile.out main.py no_plot")
os.system("wget -O gprof2dot.py https://raw.githubusercontent.com/jrfonseca/gprof2dot/master/gprof2dot.py")
os.system(f"python gprof2dot.py -f pstats main_profile.out | dot -Tpng -o main_profile.png")
Image.open('main_profile.png').show()
os.system("rm main_profile.png main_profile.out gprof2dot.py")
