

######## imports for the clients, i.e., python scripts or jupyter notebooks

import pandas as pd  
import math
import numpy as np
import scipy.stats as stats
from datetime import datetime
import random
from time import perf_counter 
import statsmodels.api as sm
from datetime import datetime
#random.seed(1)

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# notebook display
# https://github.com/ipython/ipykernel/issues/231
#%matplotlib notebook                             

# Insert path to core at front of python path
import sys
import os
# path_to_core = '../../server/core_code'                               # needs to be changed if we change file structure
# sys.path.insert(0, path_to_core); 
#print(sys.path)

### our functions
# import database_functions as dbf
# import functions_preproc as prp
# import functions_proc as pr
# import functions_postproc as pop

# database driver
import pymongo
from bson import ObjectId
from pprint import pprint
