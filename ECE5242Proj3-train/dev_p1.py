from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import load_data
import matplotlib
import MapUtils.MapUtils as utils
import MapUtilsCython.MapUtils_fclad as cutils
import test_data
from datetime import datetime
import sys

matplotlib.use("macosx")
sys.setrecursionlimit(6000)
paths = "Encoders20 Encoders21 Encoders23 Encoders22 Encoders24 Hokuyo20 Hokuyo21 " \
        "Hokuyo23 Hokuyo22 Hokuyo24".split(' ')


class Motion_Model(pd.DataFrame, object):
    def __new__(cls, name: str):
        cls.inner_width = 311.15 / 10
        cls.outer_width = 476.25 / 10
        cls.center_width = (cls.inner_width + cls.outer_width) / 2
        cls.wheel_radius = ((584.2 - 330.2) / 2) / 10
        cls.circum = 2 * cls.wheel_radius * np.pi
        cls.counts_rev = 360
        return object.__new__(cls)

    def __init__(self, name: str):
        print("init")
        super(Motion_Model, self).__init__()
        assert type(name) == str
        assert '.' not in name
        path = './data/' + name
        data = load_data.get_encoder(path)
        for i, w in enumerate(['front_right', 'front_left', 'rear_right', 'rear_left', 'time']):
            self[w] = data[i]
        R, L = self['front_right'], self['front_left']
        T = self.delta_rotation_theta(L, R)
        self['d_theta'] = T
        self['d_xy'] = self.delta_local_coord(L, R, T)


    def delta_rotation_theta(self, left, right):
        # Q is the lecure wrong in saying to just use ticks?
        r = (right / self.counts_rev) * self.circum
        l = (left / self.counts_rev) * self.circum
        return (r - l) / self.center_width

    def delta_local_coord(self, left, right, d_theta):
        # Q this seems incorrect....cos(t) should give x, not sin, and also simply the delta of
        #  theta does not give you the actual delta in coordinates...
        # returning x, y
        r = (right / self.counts_rev) * self.circum
        l = (left / self.counts_rev) * self.circum
        c = (l + r) / 2
        return list(zip(
            c * np.cos(d_theta / 2),  # x
            c * np.sin(d_theta / 2)  # y
        ))

    def dlocal_to_dglobal(self, theta: float, dx: float, dy: float):
        # consider moving theta/2 to inside funciton..
        # basis change
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        c = np.array([[dx],[dy]])
        return tuple((rotation  @ c).flatten())

    def delta_translation(self, prev: tuple, next: tuple):
        # returns distance traveled
        assert type(next) == tuple, next
        assert type(prev) == tuple, prev
        return np.sqrt((prev[0] - next[0]) ** 2 + (prev[1] - next[1]) ** 2)

    def delta_rotation_1(self, prev, next):
        return np.arctan2(next[1] - prev[1], next[0] - prev[0])

    def delta_rotation_2(self, prev_theta, next_theta, d_rot_1):
        return next_theta - prev_theta - d_rot_1

    def find_path(self, df, ind=0):
        """put in list with first value"""
        assert type(df) == list
        dxy = self['d_xy'][ind]
        dt = self['d_theta'][ind]
        dt = dt / 2
        if ind == 0:
            dxy = self.dlocal_to_dglobal(dt, dxy[0], dxy[1])
            new = pd.Series([dxy, dt], index=['xy', 'theta']) # orientaiton
            df.append(new)
            return self.find_path(df, ind + 1)
        _xy = df[ind - 1][0]
        _t = df[ind - 1][1]

        t = _t + dt
        dxy = self.dlocal_to_dglobal(t, dxy[0], dxy[1])
        new = pd.Series(
            [(_xy[0] + dxy[0], _xy[1] + dxy[1]), t],
            index=['xy', 'theta']
        )
        df.append(new)
        if ind + 1 == len(self):
            return pd.DataFrame(df)
        # print(ind, len(self))
        return self.find_path(df, ind + 1)

    def find_component_delta(self):
        df = list()
        for i in range(0, len(self)):
            xy = self['xy'][i]
            t = self['theta'][i]
            if i == 0:
                _xy = (0, 0)
                _t = 0
            else:
                _xy = self['xy'][i - 1]
                _t = self['theta'][i - 1]
            trans = self.delta_translation(_xy, xy)
            rot1 = self.delta_rotation_1(_xy, xy)
            rot2 = self.delta_rotation_2(_t, t, rot1)
            df.append(pd.Series(
                [trans, rot1, rot2],
                index=['d_trans', 'd_rot1', 'd_rot2']
            ))
        return pd.DataFrame(df)

    def process_motion(self):
        x = self.find_path(list())
        # self.merge(x,how='outer', left_index=True, right_index=True, in_place=True)
        self['xy'] = x['xy']
        self['theta'] = x['theta']
        y = self.find_component_delta()
        self['d_trans'] = y['d_trans']
        self['d_rot1'] = y['d_rot1']
        self['d_rot2'] = y['d_rot2']

    def plot_path(self):
        s = self[['xy']].explode('xy')
        s.index.name = 'i'
        s['c'] = s.groupby('i').cumcount()
        s = s.reset_index().pivot(columns='c', index='i', values='xy')
        plt.plot(s[0], s[1])
        plt.show()


class Observation_Model(object):
    def __init__(self, name, motion, lo_occ=1, lo_free=-1, cell_scale_cm = 20,
                 plot_live=False, use_cython=False, diagnostic = False):
        # lidar coming in as meters, motion coming in as mm, converting both to cm
        lidar_dict = load_data.get_lidar("./data/" + name)
        li_df = pd.DataFrame(lidar_dict)
        self.diagnostic = diagnostic
        self.plot_live = plot_live
        self.cy = use_cython
        self.scale = cell_scale_cm
        self.scan = li_df['scan'].to_list()
        self.scan = pd.DataFrame(self.scan)
        self.scan = self.scan * 100 # m to cm
        self.angles = li_df['angle'].to_list()
        self.angles = pd.DataFrame([i.ravel() for i in self.angles])
        self.time_stamp = li_df['t']
        self.motion = motion

        ## todo wrap into motion class
        s = self.motion[['xy']].explode('xy')
        s.index.name = 'i'
        s['c'] = s.groupby('i').cumcount()
        s = s.reset_index().pivot(columns='c', index='i', values='xy')
        self.motion['x'] = s[0]
        self.motion['y'] = s[1]
        ##
        if len(self.time_stamp) > len(self.motion):
            print(self.motion)
            self.time_stamp = self.motion["time"]


        # Grid size Calculation
        w = (self.motion['x'].max() - self.motion['x'].min())
        h = (self.motion['y'].max() - self.motion['y'].min())
        _w = int((w  * 3)/ self.scale)
        _h = int((h  * 3)/ self.scale)
        u = max(_w,_h)
        self.grid = np.zeros((u, u))
        self.path_grid = self.grid.copy()

        # offset calculation
        mx = self.motion['x'].mean() / self.scale
        my = self.motion['y'].mean() / self.scale
        o_x = - mx + _w / 2
        o_y = - my + _h / 2
        self.offset = np.array([[int(o_x)], [int(o_y)]])

        if self.diagnostic:
            self.lo_occ = 25
        else:
            self.lo_occ = lo_occ
        self.lo_free = lo_free



    def global_to_grid(self, coord): # how many cm in a cell,
        """assumes first column/row is x and the other y"""
        grid_cm = self.scale
        global_cm = 1 # how many cm in a global unit (converted to cm in initialization)
        if type(coord) != np.ndarray:
            coord = np.array(coord)
        assert 2 in coord.shape, coord.shape
        if coord.ndim == 1:
            coord = coord[:,np.newaxis]
        assert coord.ndim == 2, coord.shape
        if not coord.shape[0] == 2:
            coord = coord.T

        coord = coord * (1 / grid_cm) * (global_cm / 1)
        coord = coord.astype("float32")
        coord = np.round(coord, 0)

        offset_coord = coord + self.offset
        assert offset_coord.all() > 0
        assert offset_coord.all() < len(self.grid)

        return offset_coord.astype("short") # offset for negative indices
        # cm to grid cells....
        # scale of 1 cell = 1 cm


    def build_map(self):
        coord_robot = self.motion[['x', 'y']].to_numpy()
        # coord_robot = self.global_to_grid(coord_robot)
        gx_r = coord_robot[:, 0].flatten()
        gy_r = coord_robot[:, 1].flatten()
        lid_dist = self.scan.to_numpy().astype("float32")
        lid_thetas = self.angles.to_numpy().astype("float32")
        g_theta = self.motion[["theta"]].to_numpy().astype("float32")
        if len(g_theta) < len(lid_thetas):
            lid_thetas = lid_thetas[:len(g_theta), :] + g_theta
        else:
            lid_thetas = lid_thetas + g_theta[:len(lid_thetas), :]

        fig = plt.figure(figsize=(3, 3), dpi=200)
        ax = fig.add_subplot(111)
        for i, _ in enumerate(self.time_stamp):
            x_r = gx_r[i]
            y_r = gy_r[i]
            dist = lid_dist[i].flatten()
            theta = lid_thetas[i].flatten()
            d_lid_x = dist * np.cos(theta)
            d_lid_y = dist * np.sin(theta)
            d_lid_x = d_lid_x[dist > 150] # thresholding at 1.5m min distance
            d_lid_y = d_lid_y[dist > 150]
            coord_end_lidar = (
                x_r + d_lid_x,
                y_r + d_lid_y
            )
            w = self.global_to_grid([[x_r], [y_r]]) # scaling to grid
            x_r, y_r = w[0].item(), w[1].item()
            coord_end_lidar = self.global_to_grid(coord_end_lidar)
            if self.diagnostic:
                self.grid[:] = 0
            self.grid[- coord_end_lidar[1], coord_end_lidar[0]] += self.lo_occ
            ### To use range finding, uncomment: ###
            if not self.diagnostic:
                free_cells = self.range_find(
                    x_r,
                    y_r,
                    coord_end_lidar[0],
                    coord_end_lidar[1],
                )

                self.grid[- free_cells[1], free_cells[0]] += self.lo_free

            self.grid = self.grid.clip(-50,50)
            if self.diagnostic:
                if i % 20 == 0:
                    print(i)
                    self.plot_live_frame(i, ax)
            else:
                if i % 100 == 0:
                    print(i)
                    if self.plot_live:
                        self.plot_live_frame(i, ax)


    def calc_probability(self, x):
        """Must be the log(x) value"""
        return 1 - (1 / (1 + np.exp(x)))


    def range_find(self, x_r, y_r, x_l, y_l):
        """return 2d array of coordinates of intersecting coordinates"""
        x_r = np.short(x_r)
        y_r = np.short(y_r)
        x_l = np.short(x_l)
        y_l = np.short(y_l)
        assert len(x_l) == len(y_l), [len(x_l), len(y_l)]
        if self.cy:
            return cutils.getMapCellsFromRay_fclad(x_r, y_r, x_l, y_l, 0)
        else:
            return np.int32(utils.getMapCellsFromRay(x_r, y_r, x_l, y_l, 0))


    def plot_map(self):
        fig = plt.figure(figsize=(10,10),dpi=200)
        ax = fig.add_subplot(111)
        grid = self.calc_probability(self.grid)
        ax.imshow(grid,cmap="Greys")
        plt.show()


    def plot_live_frame(self, t, ax):
        print("live plot updated")
        ax.cla()
        ax.set_title("frame {}".format(t))

        # map
        grid = self.calc_probability(self.grid)
        ax.imshow(grid, cmap="Greys")

        # path
        s = self.global_to_grid(
                self.motion[["x", "y"]].loc[:t].to_numpy()
        )
        x = self.path_grid
        x[- s[1, :], s[0, :]] = 1
        cmap1 = matplotlib.colors.ListedColormap(['none', 'red'])
        ax.imshow(x, cmap=cmap1)
        plt.pause(0.1)


    def plot_map_with_path(self, name = None):
        fig = plt.figure(figsize=(3, 3), dpi=200)
        ax = fig.add_subplot(111)

        # map
        grid = self.calc_probability(self.grid)
        ax.imshow(grid, cmap="Greys")

        # path
        s = self.global_to_grid(
            self.motion[["x", "y"]].to_numpy()
        )
        x = self.path_grid
        x[ - s[1, :], s[0, :]] = 1
        cmap1 = matplotlib.colors.ListedColormap(['none', 'green'])
        ax.imshow(x, cmap=cmap1)
        t = str(datetime.utcnow())[-5:]
        if name:
            path = f"../results/map_{t}.png"
        else:
            path = f"../results/map_{name}.png"
        plt.savefig(path)
        plt.show()


def run(j=0,k=5):
    for i in range(j,k):
        M = Motion_Model(paths[i])
        M.process_motion()
        Obs = Observation_Model(
            paths[i+5],
            M,
            lo_occ = .6,
            lo_free = -.1,
            cell_scale_cm = 20,
            use_cython = False, # DOESN'T WORK!!!
            plot_live = False,
            diagnostic= True
        )
        name = paths[i][-2:]
        Obs.build_map()
        Obs.plot_map_with_path(name)


if __name__ == "__main__":
    run(2,3)