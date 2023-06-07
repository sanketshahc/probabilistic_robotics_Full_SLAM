# # Test Path Data
# from dev import *
#
#
# def test_path_data(X):
#     if
#     X = lambda t: t[0] + np.sin(t[1])
#     for i in range(100):
#     np.sin()
#
#
#     return X(i)
import numpy as np
import pandas as pd

# circle path coords with orientation
def circle_polar(theta,r):
    return np.array(r*np.cos(theta)), np.array(r*np.sin(theta)), theta + (np.pi/2)

t = np.linspace(0,2*np.pi,num=256)
x, y, tt = circle_polar(t,4)
test_df = pd.DataFrame([zip(x,y),tt]).T
test_df.index.name = "time"
test_df.rename(columns={0 : "xy", 1 : "theta"}, inplace=True)
test_df.reset_index(inplace=True)