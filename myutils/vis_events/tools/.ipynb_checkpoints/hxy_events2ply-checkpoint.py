from numpy.lib.type_check import imag
from plyfile import PlyData, PlyElement 
import numpy as np
from scipy.optimize.optimize import vecnorm 
import cv2
import h5py
import os


def read_h5_events(filename, start_idx, end_idx, inp_prex = 'down4'):
    h5_file = h5py.File(filename, 'r')
    xs = h5_file[f'{inp_prex}_events/xs'][start_idx:end_idx]
    ys = h5_file[f'{inp_prex}_events/ys'][start_idx:end_idx]
    ts = h5_file[f'{inp_prex}_events/ts'][start_idx:end_idx]
    ps = h5_file[f'{inp_prex}_events/ps'][start_idx:end_idx]

    sensor_resolution = h5_file.attrs['sensor_resolution']

    return sensor_resolution, xs, ys, ts, ps


def main():
    filename = ''
    basename = os.path.basename(filename).split('.')[0]
    TOTAL_COUNT = 122033
    start_idx = 0

    vertices_final = np.empty(0, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    sensor_resolution, xs, ys, ts, ps = read_h5_events(filename, start_idx=start_idx, end_idx=start_idx+TOTAL_COUNT)
    # flip 
    H, W = sensor_resolution
    max = ts.max()
    min = ts.min()
    ts = (ts - min) / (max - min) * H

    # xs = 240 - xs
    # ys = 180 - ys
    # image 
    # image = np.zeros((180,240),dtype=np.uint8)
    # for i in range(TOTAL_COUNT):
    #     x, y = xs[i], ys[i]
    #     image[180-y,x-1] = 255
    # cv2.imwrite("stacking_{}.png".format(i),image)
    # LAYER_NUM = 10
    # LAYER_COUNT = TOTAL_COUNT // LAYER_NUM
    print("event count: {}, duration time {:0.2f}".format(TOTAL_COUNT, ts[-1]-ts[0]))
    # bottom_xs,bottom_ys,bottom_ts = xs[0:LAYER_COUNT],ys[0:LAYER_COUNT],ts[0:LAYER_COUNT]
    # top_xs,top_ys,top_ts = xs[int(9*LAYER_COUNT):],ys[int(9*LAYER_COUNT):],ts[int(9*LAYER_COUNT):]
    # connect the proper data structures
    vertices = np.empty(TOTAL_COUNT, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    # red = np.ones((LAYER_COUNT,))*255*(i%3==0)
    # green = np.ones((LAYER_COUNT,))*255*(i%3==1)
    # blue = np.ones((LAYER_COUNT,))*255*(i%3==2)
    red = ps*255
    green = ps*0
    blue = (ps==-1)*255
    vertices['x'] = xs.astype('f4')    
    vertices['y'] = ys.astype('f4')
    vertices['z'] = ts.astype('f4')
    vertices['red']   = red.astype('u1')
    vertices['green'] = green.astype('u1')
    vertices['blue']  = blue.astype('u1')
    vertices_final = np.concatenate((vertices_final, vertices))


    # save as ply
    ply = PlyData([PlyElement.describe(vertices_final, 'vertex')], text=False)
    ply.write(f'/disk/work/output/dataset_ply/{basename}.ply')


if __name__ == '__main__':
    main()