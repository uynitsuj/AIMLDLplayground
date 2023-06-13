import numpy as np
from linalgutils import *
from pyqtgraph.Qt import QtCore, QtWidgets
from multiprocessing import Process, shared_memory
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import sys
import time


class Visualizer(object):
    def __init__(self):
        self.pose = []
        self.samplesphere = []
        self.app = QtWidgets.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.orbit(45, 1)
        self.w.opts['distance'] = 7
        self.w.setWindowTitle('pyqtgraph Pose')
        self.w.setGeometry(1000, 500, 800, 500)
        self.w.show()
        self.setup()

    def setup(self):
        gsz = 1
        gsp = .1
        gx = gl.GLGridItem(color=(255, 255, 255, 60))
        gx.setSize(gsz, gsz, gsz)
        gx.setSpacing(gsp, gsp, gsp)
        gx.rotate(90, 0, 1, 0)
        # gx.translate(-gsz/2, 0, gsz/2)
        self.w.addItem(gx)
        gy = gl.GLGridItem(color=(255, 255, 255, 60))
        gy.setSize(gsz, gsz, gsz)
        gy.setSpacing(gsp, gsp, gsp)
        gy.rotate(90, 1, 0, 0)
        # gy.translate(0, -gsz/2, gsz/2)
        self.w.addItem(gy)
        gz = gl.GLGridItem(color=(255, 255, 255, 100))
        gz.setSize(gsz, gsz, gsz)
        gz.setSpacing(gsp, gsp, gsp)
        self.w.addItem(gz)
        self.w.setBackgroundColor('black')

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec_()

    def update(self):
        del self.w.items[:]
        self.w.clear()
        self.setup()
        self.w.opts['azimuth'] += 0.25

        quatdisp = np.ndarray((7,), dtype=np.float64)
        desiredpose = np.ndarray(
            (7,), dtype=np.float64, buffer=self.pose.buf)
        quatdisp = desiredpose

        if quatdisp[0]:
            pose = quaternion_to_se3(normalize_quaternion(
                (quatdisp[0], quatdisp[1], quatdisp[2], quatdisp[3])))
            pose[0][3] = quatdisp[4]
            pose[1][3] = quatdisp[5]
            pose[2][3] = quatdisp[6]

            width = 10
            w = [pose[0][3], pose[1][3], pose[2][3]]
            v1 = [pose[0][0], pose[1][0], pose[2][0]]
            v2 = [pose[0][1], pose[1][1], pose[2][1]]
            v3 = [pose[0][2], pose[1][2], pose[2][2]]
            v1 = np.append([w], [np.add(w, v1)], axis=0)*0.6

            v2 = np.append([w], [np.add(w, v2)], axis=0)*2

            v3 = np.append([w], [np.add(w, v3)], axis=0)

            self.w.addItem(gl.GLLinePlotItem(
                pos=v1, color=(1.0, 0.0, 0.0, 1.0), width=width, antialias=True))
            self.w.addItem(gl.GLLinePlotItem(
                pos=v2, color=(0.0, 1.0, 0.0, 1.0), width=width, antialias=True))
            self.w.addItem(gl.GLLinePlotItem(
                pos=v3, color=(0.0, 0.0, 1.0, 1.0), width=width, antialias=True))

            poseso3 = quaternion_to_so3(normalize_quaternion(
                (quatdisp[0], quatdisp[1], quatdisp[2], quatdisp[3])))
            poseso3[:, 0] *= 0.6
            poseso3[:, 1] *= 2
            altsphere = []
            for point in self.samplesphere:
                pointnp = np.asarray(point)
                altsphere.append(poseso3@(pointnp))
            self.w.addItem(gl.GLScatterPlotItem(
                pos=altsphere, color=(0.1, 0.3, 0.5, 0.99), size=10))

    def animation(self):
        self.pose = shared_memory.SharedMemory(name='pose')
        self.samplesphere = sample_unit_sphere(300, 300, 300)
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(50)
        self.start()


def rotate_z(shared_name, shape):
    existing_shm = shared_memory.SharedMemory(name=shared_name)
    rot_z = np.array([[np.cos(np.deg2rad(.5)), -np.sin(np.deg2rad(.5)), 0],
                      [np.sin(np.deg2rad(.5)), np.cos(np.deg2rad(.5)), 0],
                      [0, 0, 1]])

    while True:
        pose = np.ndarray(shape, dtype=np.float64, buffer=existing_shm.buf)
        poseso3 = orthonormalize_matrix(quaternion_to_so3(
            np.array([pose[0], pose[1], pose[2], pose[3]])))
        pose[:] = np.append(so3_to_quaternion(poseso3 @ rot_z), [0, 0, 0])
        time.sleep(10/1000)


def main():
    try:
        poseshm = shared_memory.SharedMemory(name='pose',
                                             create=True, size=8)
    except:
        print("Obliterating existing shared memory")
        poseshm = shared_memory.SharedMemory(name='pose',
                                             create=False, size=8)
        poseshm.close()
        poseshm.unlink()
    v = Visualizer()
    initpose = np.identity(3)
    posequat = np.append(so3_to_quaternion(initpose), [0, 0, 0])
    buffer = np.ndarray((7,),
                        dtype=np.float64, buffer=poseshm.buf)
    buffer[:] = posequat[:]
    rotateframe = Process(target=rotate_z, args=(poseshm.name, posequat.shape))
    rotateframe.start()
    try:
        v.animation()
    except Exception as e:
        print(e)
    finally:
        try:
            print("Exiting...")
            v.app.quit()
            poseshm.close()
            poseshm.unlink()
        finally:
            sys.exit()


if __name__ == "__main__":
    main()
