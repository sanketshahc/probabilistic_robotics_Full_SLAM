from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import load_data
import matplotlib
import MapUtils.MapUtils as utils
import MapUtilsCython.MapUtils_fclad as cutils
import random
import scipy.special as sp
import test_data
from datetime import datetime
import sys

matplotlib.use("macosx")
file_names = {
    1: ("Encoders20", "Hokuyo20"),
    2: ("Encoders21", "Hokuyo21"),
    3: ("Encoders23", "Hokuyo23"),
    4: ("Encoders22", "Hokuyo22"),
    5: ("Encoders24", "Hokuyo24")
}
hyper_params = {
        "robot_width" : 75,
        "alpha_scale": .002,
        "num_particles" : 1,
        "lo_occ" : .6,
        "lo_free" : -.2,
        "cell_scale_cm": 20,
        "use_cython" : False,  # DOESN'T WORK!!!
        "plot_live" : True,
        "diagnostic": False,
        "disable_noise": True,
        "disable_range_find": False,
        "threshold": 150, #cm
        "log_clip": 50
    }


class Particle_Filter(object):
    def __init__(self, n: int, names: tuple):
        self.log_from_prob = lambda x: - 1 / np.log(x)
        self.prob_from_log = lambda x: np.exp(-1/x)
        self.particles = {0: [Robot(names[0], is_main=True),self.log_from_prob(1/n)]}
        self.particles.update({i: [Robot(names[0]), self.log_from_prob(1/n)] for i in range(1,n)})
        self.diagnostic = hyper_params["diagnostic"]
        self.best_path = pd.DataFrame(
            {
                'x': [],
                'y': [],
                'global_theta': []
            }
        )
        self.plot_live = hyper_params["plot_live"]
        self.environment = Observation_Model(
            names[1],
            lo_occ=hyper_params["lo_occ"],
            lo_free=hyper_params["lo_free"],
            cell_scale_cm=hyper_params["cell_scale_cm"],
            use_cython=hyper_params["use_cython"],
            diagnostic=hyper_params["diagnostic"]
        )
        length = min(len(self.environment.time_stamp), len(self.particles[0][0]["time"]))
        clock.set_length(length)


    def resample(self):
        #   from weighted objects, ensure weights are normalized (sum to 1)
        robots_weights = self.particles.values()
        particles = [i[0] for i in robots_weights]
        log_weights = [i[1] for i in robots_weights]
        draws = [random.uniform(0, 1) for _ in robots_weights]
        if not self.diagnostic:
            if len(log_weights) > 1:
                weights = [self.prob_from_log(w) for w in log_weights]
                weights = np.array(weights) / np.array(weights).sum()
                assert np.allclose(
                    np.array(weights).sum() , 1.
                ), np.array(weights).sum()
            else:
                weights = [1]
        else:
            weights = [1]
        intervals = list()
        p = 0
        for w in weights:
            intervals.append(pd.Interval(p, p + w))
            p += w

        for d, draw in enumerate(draws):
            for i, interval in enumerate(intervals):
                if draw in interval:
                    self.particles.update(
                        {d: [particles[i], self.log_from_prob(1/len(particles))]})


    def evaluate_log_weights(self):
        """returns weight vector"""
        if clock.time > 0:
            for (i, rw) in self.particles.items():
                robot = rw[0]
                coord = robot[["x","y","global_theta"]].loc[clock.time]
                self.environment.update_robot_coord(coord)
                log_weight = self.environment.evaluate_observation_fit()
                self.particles[i][1] = log_weight

    def _reset_log_weights(self):
        for (i, rw) in self.particles.items():
            self.particles[i][1] = 1 / len(self.particles)  # should keys should be index

    def update_best_path(self):
        i = clock.time
        weights = [rw[1] for rw in self.particles.values()]
        weights = np.array(weights)
        m = np.argmax(weights)
        best_robot = self.particles[m][0]
        coord = best_robot[["x","y","global_theta"]].loc[i]
        x, y, t = coord
        self.best_path.loc[i,"x"] = x
        self.best_path.loc[i,"y"] = y
        self.best_path.loc[i,"global_theta"] = t
        self.environment.update_robot_coord(coord)

    def update_environment_map(self):
        self.environment.update_map()

    def propose_motions(self):
        for (i, rw) in self.particles.items():
            robot = rw[0]
            robot.path_step()

    def slam(self):
        fig = plt.figure(figsize=(3, 3), dpi=200)
        ax = fig.add_subplot(111)
        print("new ax")
        for i in clock:
            self.propose_motions()
            self.evaluate_log_weights()
            self.resample()
            self.update_best_path()
            self.update_environment_map()
            # Reporting:

            if self.diagnostic:
                if i % 20 == 0:
                    print(i)
                    self.plot_live_frame(ax)
                self.environment.reset_map()
            else:
                if i % 100 == 0:
                    print(i)
                    if self.plot_live:
                        self.plot_live_frame(ax)

    def plot_live_frame(self, ax):
        """assuming already up until current time."""
        print("live plot updated")
        ax.cla()
        ax.set_title("frame {}".format(clock.time))

        # map
        grid = self.environment.calc_probability(self.environment.grid)
        ax.imshow(grid, cmap="Greys")

        # path
        s = self.environment.global_to_grid(
            self.best_path[["x", "y"]].to_numpy()
        )
        x = self.environment.path_grid
        x[- s[1, :], s[0, :]] = 1
        cmap1 = matplotlib.colors.ListedColormap(['none', 'red'])
        ax.imshow(x, cmap=cmap1)
        plt.pause(0.1)

    def plot_map_with_path(self, name =None):
        fig = plt.figure(figsize=(6, 6), dpi=200)
        ax = fig.add_subplot(111)
        # map
        grid = self.environment.calc_probability(self.environment.grid)
        ax.imshow(grid, cmap="Greys")

        # path
        s = self.environment.global_to_grid(
            self.best_path[["x", "y"]].to_numpy()
        )
        x = self.environment.path_grid
        x[- s[1, :], s[0, :]] = 1
        cmap1 = matplotlib.colors.ListedColormap(['none', 'green'])
        ax.imshow(x, cmap=cmap1)
        t = str(datetime.utcnow())[-5:]
        if not name:
            path = f"../results/map_{t}.png"
        else:
            path = f"../results/map_{name[0]}.png"
        plt.savefig(path)


class Robot(pd.DataFrame, object):
    def __new__(cls, name: str, is_main = False):
        width = hyper_params["robot_width"]
        cls.inner_width = 311.15 / 10
        cls.outer_width = 476.25 / 10
        if not width:
            cls.center_width = (cls.inner_width + cls.outer_width) / 2  # should make this a parameter
        else:
            cls.center_width = width
        cls.wheel_radius = ((584.2 - 330.2) / 2) / 10
        cls.circum = (2 * cls.wheel_radius * np.pi)
        cls.counts_rev = 360

        return object.__new__(cls)

    def __init__(self, name: str, is_main = False):
        print("robot initialized")
        super(Robot, self).__init__()
        assert type(name) == str
        assert '.' not in name
        path = './data/' + name
        data = load_data.get_encoder(path)
        for i, w in enumerate(['front_right', 'front_left', 'rear_right', 'rear_left', 'time']):
            self[w] = data[i]
        R, L = self['front_right'], self['front_left']
        T = self.delta_rotation_theta(L, R)
        self['d_locrot_theta'] = T
        self['d_local_xy'] = self.delta_local_coord(L, R, T)
        self['x'] = None
        self['y'] = None
        self['global_theta'] = None
        if is_main or hyper_params["disable_noise"]:
            self.a_1 = 0
            self.a_2 = 0
            self.a_3 = 0
            self.a_4 = 0
        else:
            self.a_1 = random.uniform(0, 1) * hyper_params["alpha_scale"]
            self.a_2 = random.uniform(0, 1) * hyper_params["alpha_scale"]
            self.a_3 = random.uniform(0, 1) * hyper_params["alpha_scale"]
            self.a_4 = random.uniform(0, 1) * hyper_params["alpha_scale"]

    def delta_rotation_theta(self, left, right):
        r = (right / self.counts_rev) * self.circum
        l = (left / self.counts_rev) * self.circum
        return (r - l) / self.center_width

    def delta_local_coord(self, left, right, d_theta):
        # d_theta is NOT d_locrot_theta, it is 1/2 of it
        # returning x, y
        r = (right / self.counts_rev) * self.circum
        l = (left / self.counts_rev) * self.circum
        c = (l + r) / 2
        return list(zip(
            c * np.cos(d_theta / 2),  # x
            c * np.sin(d_theta / 2)  # y
        ))

    def delta_translation(self, prev: tuple, next: tuple):
        # returns distance traveled
        assert type(next) == tuple, next
        assert type(prev) == tuple, prev
        return np.sqrt((prev[0] - next[0]) ** 2 + (prev[1] - next[1]) ** 2)


    def delta_rotation_1(self, prev, next):
        return np.arctan2(next[1] - prev[1], next[0] - prev[0])


    def delta_rotation_2(self, prev_theta, next_theta, d_rot_1):
        return next_theta - prev_theta - d_rot_1


    def dlocal_to_dglobal(self, theta: float, dx: float, dy: float):
        # consider moving theta/2 to inside funciton..
        # basis change
        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        c = np.array([[dx], [dy]])
        return tuple((rotation @ c).flatten())


    def disturb_components(self, components):
        """trans, rot1, rot2"""
        trans, rot1, rot2 = components
        n = lambda x: random.gauss(0, x)
        trans_ = trans + n(
            (self.a_3 * abs(trans)) + (self.a_4 * (abs(rot1) + abs(rot2)))
        )
        rot1_ = rot1 + n(
            (self.a_1 * abs(rot1)) + (self.a_2 * abs(trans))
        )
        rot2_ = rot2 + n(
            (self.a_1 * abs(rot2)) + (self.a_2 * abs(trans))
        )
        return trans_, rot1_, rot2_
        # e_trans =

    def coord_from_components(self, coord, components):
        x, y, t = coord
        trans, rot1, rot2 = components
        return (
            x + (trans * np.cos(t + rot1)),
            y + (trans * np.sin(t + rot2)),
            t + rot1 + rot2
        )

    def get_current_coords(self):
        i = clock.time
        coord = tuple(self[["x", "y", "global_theta"]].loc[i])
        if coord[0] == None:
            return (0,0,0)
        else:
            return coord


    def path_step(self):
        """ df is a list of Series. put in list with first value"""
        ind = clock.time
        # assert type(df) == list
        # take df from self

        dxy = self['d_local_xy'][ind]
        dt = self['d_locrot_theta'][ind]  # this is the rotation theta
          # this is the LOCAL / Global orientation change (1/2 the rotaiton theta)

        if ind == 0:
            dx, dy = self.dlocal_to_dglobal(dt / 2, dxy[0], dxy[1])
            self.loc[0,"x"] = dx
            self.loc[0,"y"] = dy
            self.loc[0, "global_theta"] = dt

        else:
            _x = self.loc[ind - 1]["x"]
            _y = self.loc[ind - 1]["y"]
            _t = self.loc[ind - 1]["global_theta"]

            ####Quick Patch####
            _x = 0 if _x == None else _x
            _y = 0 if _y == None else _y
            _t = 0 if _t == None else _t
            ######

            t = _t + dt / 2
            dx, dy = self.dlocal_to_dglobal(t, dxy[0], dxy[1])
            x = _x + dx
            y = _y + dy
            trans = self.delta_translation((_x, _y), (x, y))
            rot1 = self.delta_rotation_1((_x, _y), (x, y))
            rot2 = self.delta_rotation_2(_t, t, rot1)
            components = trans, rot1, rot2
            coord = x, y, t
            components = self.disturb_components(components)
            nx, ny, nt = self.coord_from_components(coord, components)
            ntrans, nrot1, nrot2 = components
            self.loc[ind, "x"] = nx
            self.loc[ind, "y"] = ny
            self.loc[ind,"global_theta"] = nt
            self.loc[ind,'d_trans'] = ntrans
            self.loc[ind,'d_rot1'] = nrot1
            self.loc[ind,'d_rot2'] = nrot2

    # todo vefify
    def plot_path(self):
        plt.plot(self['x'], self['y'])
        plt.show()


class Observation_Model(object):
    def __init__(self,
                 name,
                 lo_occ=1,
                 lo_free=-1,
                 cell_scale_cm = 20,
                 plot_live=False,
                 use_cython=False,
                 diagnostic = False
                 ):
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
        self.robot_coord = (0,0,0)

        w = 2000
        h = 2000
        _w = int((w  * 5)/ self.scale)
        _h = int((h  * 5)/ self.scale)
        u = max(_w,_h)
        self.grid = np.zeros((u, u))
        self.path_grid = self.grid.copy()
        o_x =  _w / 2
        o_y =  _h / 2
        self.offset = np.array([[int(o_x)], [int(o_y)]])

        if self.diagnostic:
            self.lo_occ = 25
        else:
            self.lo_occ = lo_occ
        self.lo_free = lo_free

    def update_robot_coord(self, coord):
        assert type(coord) == pd.Series
        c = tuple(coord)
        assert len(c) == 3
        assert None not in coord, coord
        self.robot_coord = c

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

        if not self.diagnostic:
            assert offset_coord.all() > 0
            assert offset_coord.all() < len(self.grid)

        return offset_coord.astype("short") # offset for negative indices
        # cm to grid cells....
        # scale of 1 cell = 1 cm

    def get_current_lid_g_coords(self):
        i = clock.time
        gx_r, gy_r, g_theta = self.robot_coord # tuple or series?
        lid_dist = self.scan.to_numpy().astype("float32")  # is 2d
        lid_thetas = self.angles.to_numpy().astype("float32")  # is 2d
        assert i < len(lid_thetas), "Ensure correct time length in main function"
        cur_lid_thetas = lid_thetas[i].flatten() # -> 1d array
        lid_g_thetas = cur_lid_thetas + g_theta
        # no longer adding entire theta vector, just add  array to scalar
        dist = lid_dist[i].flatten() # -> 1d array
        return self.lid_g_coords_from_robot_g_coords((gx_r,gy_r), dist, lid_g_thetas)

    def lid_g_coords_from_robot_g_coords(self, robot_xy, dist, thetas):
        """from roobot xy, dist, and lidar global thetas"""
        assert type(robot_xy) == tuple
        x_r, y_r = robot_xy[0], robot_xy[1]
        d_lid_x = dist * np.cos(thetas)
        d_lid_y = dist * np.sin(thetas)
        d_lid_x = d_lid_x[dist > hyper_params["threshold"]]  # thresholding at 1.5m min distance
        d_lid_y = d_lid_y[dist > hyper_params["threshold"]]
        coord_end_lidar = (
            x_r + d_lid_x,
            y_r + d_lid_y
        )
        return coord_end_lidar

    def reset_map(self):
        self.grid[:] = 0

    def update_map(self):
        i = clock.time
        gx_r, gy_r, g_theta = self.robot_coord
        coord_end_lidar = self.get_current_lid_g_coords()
        xy = self.global_to_grid([[gx_r], [gy_r]]) # scaling to grid
        x_r, y_r = xy[0].item(), xy[1].item()
        coord_end_lidar = self.global_to_grid(coord_end_lidar)
        self.grid = self.grid.clip(-hyper_params["log_clip"],hyper_params["log_clip"])
        self.grid[- coord_end_lidar[1], coord_end_lidar[0]] += self.lo_occ
        if not self.diagnostic:
            if not hyper_params["disable_range_find"]:
                free_cells = self.range_find(
                    x_r,
                    y_r,
                    coord_end_lidar[0],
                    coord_end_lidar[1],
                )
                self.grid[- free_cells[1], free_cells[0]] += self.lo_free

    def calc_probability(self, x):
        """Must be the log(x) value"""
        return 1 - (1 / (1 + np.exp(x)))

    def evaluate_observation_fit(self, ):
        coord_end_lidar = self.get_current_lid_g_coords()
        coord_end_lidar = self.global_to_grid(coord_end_lidar)
        g = self.grid[- coord_end_lidar[1], coord_end_lidar[0]]
        s = float(g.sum())
        assert type(s) == float, type(s)
        return s

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


class Clock(object):
    def __init__(self, length = 0):
        self.time = - 1
        self.end = length - 1

    def __iter__(self):
        return self._timestep()

    def _timestep(self):
        while self.time < self.end:
            self.time += 1
            yield self.time

    def set_length(self, length):
        self.end = length - 1

def run(files:list=None):
    for i in files:
        names = file_names[i]
        print(f"running {names}")
        particle_filter = Particle_Filter(hyper_params["num_particles"], names)
        particle_filter.slam()
        particle_filter.plot_map_with_path(names[0][-3:])

    return particle_filter

if __name__ == "__main__":
    clock = Clock()
    model = run([5])
