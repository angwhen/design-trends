import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from time import time
from collections import Counter
from functools import reduce
import pickle, math, random, cv2, colorsys
import HSVHelpers, GetPixelsHelpers, YearlyColorGrids, ClusterImagesByColor

CLusterImagesByColor.make_color_clusters(Q=5, K=7, color_rep="hsv",remove_monochrome=True, remove_predom_faces = False, remove_skin=False)
YearlyColorGrids.yearly_grids(num_dom_colors=10, Q= 25, color_rep="hsv",remove_monochrome=True, remove_predom_faces = False, remove_skin=False)
