from pathlib import Path
path = Path(__file__).parents[1]/'videos/variance/'
path.mkdir(parents=True, exist_ok=True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation