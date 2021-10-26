"""
Let's draw the balls!
"""

import sys
import ctypes as ct
import numpy as np
import cv2

SHOWSZ = 800
MOUSEX, MOUSEY = 0.5, 0.5
ZOOM = 1.0
CHANGED = True
IDX = 0

def onmouse(*args):
    """
    I don't know what is it
    """
    global MOUSEX, MOUSEY, CHANGED
    y = args[1]
    x = args[2]
    MOUSEX = x / float(SHOWSZ)
    MOUSEY = y / float(SHOWSZ)
    CHANGED = True


cv2.namedWindow("show3d")
cv2.moveWindow("show3d", 0, 0)
cv2.setMouseCallback("show3d", onmouse)

dll = np.ctypeslib.load_library("render_balls_so.so", ".")


def showpoints(
    xyz,
    c_gt=None,
    c_pred=None,
    waittime=0,
    showrot=False,
    magnify_blue=0,
    freezerot=False,
    background=(0, 0, 0),
    normalizecolor=True,
    ballradius=10,
):
    """
    Function for drawing
    """
    global SHOWSZ, CHANGED, IDX

    if len(xyz.shape) == 2:
        xyz = np.expand_dims(xyz, 0)

    num_samples = xyz.shape[0]

    for i in range(num_samples):
        xyz[i] = xyz[i] - xyz[i].mean(axis=0)
        radius = ((xyz[i] ** 2).sum(axis=-1) ** 0.5).max()
        xyz[i] /= (radius * 2.2) / SHOWSZ
    if c_gt is None:
        c_0 = np.zeros((len(xyz[IDX]),), dtype="float32") + 255
        c_1 = np.zeros((len(xyz[IDX]),), dtype="float32") + 255
        c_2 = np.zeros((len(xyz[IDX]),), dtype="float32") + 255
    else:
        c_0 = c_gt[:, 0]
        c_1 = c_gt[:, 1]
        c_2 = c_gt[:, 2]

    if normalizecolor:
        c_0 /= (c_0.max() + 1e-14) / 255.0
        c_1 /= (c_1.max() + 1e-14) / 255.0
        c_2 /= (c_2.max() + 1e-14) / 255.0

    c_0 = np.require(c_0, "float32", "C")
    c_1 = np.require(c_1, "float32", "C")
    c_2 = np.require(c_2, "float32", "C")

    show = np.zeros((SHOWSZ, SHOWSZ, 3), dtype="uint8")

    def render():
        """
        render function
        """
        global ZOOM

        rotmat = np.eye(3)
        if not freezerot:
            xangle = (MOUSEY - 0.5) * np.pi * 1.2
        else:
            xangle = 0
        rotmat = rotmat.dot(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(xangle), -np.sin(xangle)],
                    [0.0, np.sin(xangle), np.cos(xangle)],
                ]
            )
        )
        if not freezerot:
            yangle = (MOUSEX - 0.5) * np.pi * 1.2
        else:
            yangle = 0
        rotmat = rotmat.dot(
            np.array(
                [
                    [np.cos(yangle), 0.0, -np.sin(yangle)],
                    [0.0, 1.0, 0.0],
                    [np.sin(yangle), 0.0, np.cos(yangle)],
                ]
            )
        )
        rotmat *= ZOOM
        nxyz = xyz[IDX].dot(rotmat) + [SHOWSZ / 2, SHOWSZ / 2, 0]

        ixyz = nxyz.astype("int32")
        show[:] = background
        dll.render_ball(
            ct.c_int(show.shape[0]),
            ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p),
            ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p),
            c_0.ctypes.data_as(ct.c_void_p),
            c_1.ctypes.data_as(ct.c_void_p),
            c_2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius),
        )

        if magnify_blue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=0))
            if magnify_blue >= 2:
                show[:, :, 0] = np.maximum(
                    show[:, :, 0], np.roll(show[:, :, 0], -1, axis=0)
                )
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=1))
            if magnify_blue >= 2:
                show[:, :, 0] = np.maximum(
                    show[:, :, 0], np.roll(show[:, :, 0], -1, axis=1)
                )
        if showrot:
            cv2.putText(
                show,
                f"xangle {int(xangle / np.pi * 180)}",
                (30, SHOWSZ - 30),
                0,
                0.5,
                cv2.cv.CV_RGB(255, 0, 0),
            )
            cv2.putText(
                show,
                f"yangle {int(yangle / np.pi * 180)}",
                (30, SHOWSZ - 50),
                0,
                0.5,
                cv2.cv.CV_RGB(255, 0, 0),
            )
            cv2.putText(
                show,
                f"ZOOM {int(ZOOM * 100)}%",
                (30, SHOWSZ - 70),
                0,
                0.5,
                cv2.cv.CV_RGB(255, 0, 0),
            )

    CHANGED = True
    while True:
        if CHANGED:
            render()
            CHANGED = False
        cv2.imshow("show3d", show)
        if waittime == 0:
            cmd = cv2.waitKey(10) % 256
        else:
            cmd = cv2.waitKey(waittime) % 256
        if cmd == ord("q"):
            break
        elif cmd == ord("Q"):
            sys.exit(0)

        if cmd == ord("t") or cmd == ord("p"):
            if cmd == ord("t"):
                if c_gt is None:
                    c_0 = np.zeros((len(xyz[IDX]),), dtype="float32") + 255
                    c_1 = np.zeros((len(xyz[IDX]),), dtype="float32") + 255
                    c_2 = np.zeros((len(xyz[IDX]),), dtype="float32") + 255
                else:
                    c_0 = c_gt[:, 0]
                    c_1 = c_gt[:, 1]
                    c_2 = c_gt[:, 2]
            else:
                if c_pred is None:
                    c_0 = np.zeros((len(xyz[IDX]),), dtype="float32") + 255
                    c_1 = np.zeros((len(xyz[IDX]),), dtype="float32") + 255
                    c_2 = np.zeros((len(xyz[IDX]),), dtype="float32") + 255
                else:
                    c_0 = c_pred[:, 0]
                    c_1 = c_pred[:, 1]
                    c_2 = c_pred[:, 2]
            if normalizecolor:
                c_0 /= (c_0.max() + 1e-14) / 255.0
                c_1 /= (c_1.max() + 1e-14) / 255.0
                c_2 /= (c_2.max() + 1e-14) / 255.0
            c_0 = np.require(c_0, "float32", "C")
            c_1 = np.require(c_1, "float32", "C")
            c_2 = np.require(c_2, "float32", "C")
            CHANGED = True

        if cmd == ord("j"):
            IDX = (IDX + 1) % num_samples
            print(IDX)
            CHANGED = True

        if cmd == ord("k"):
            IDX = (IDX - 1) % num_samples
            print(IDX)
            CHANGED = True

        if cmd == ord("n"):
            ZOOM *= 1.1
            CHANGED = True
        elif cmd == ord("m"):
            ZOOM /= 1.1
            CHANGED = True
        elif cmd == ord("r"):
            ZOOM = 1.0
            CHANGED = True
        elif cmd == ord("s"):
            cv2.imwrite("show3d.png", show)
        if waittime != 0:
            break
    return cmd


def showpoints_frame(
    xyz,
    c_gt=None,
    showrot=False,
    magnify_blue=0,
    background=(0, 0, 0),
    normalizecolor=True,
    ballradius=10,
):
    """
    Function for frame creation
    """

    if len(xyz.shape) == 2:
        xyz = np.expand_dims(xyz, 0)

    num_samples = xyz.shape[0]

    for i in range(num_samples):
        xyz[i] = xyz[i] - xyz[i].mean(axis=0)
        radius = ((xyz[i] ** 2).sum(axis=-1) ** 0.5).max()
        xyz[i] /= (radius * 2.2) / SHOWSZ
    if c_gt is None:
        c_0 = np.zeros((len(xyz[IDX]),), dtype="float32") + 255
        c_1 = np.zeros((len(xyz[IDX]),), dtype="float32") + 255
        c_2 = np.zeros((len(xyz[IDX]),), dtype="float32") + 255
    else:
        c_0 = c_gt[:, 0]
        c_1 = c_gt[:, 1]
        c_2 = c_gt[:, 2]

    if normalizecolor:
        c_0 /= (c_0.max() + 1e-14) / 255.0
        c_1 /= (c_1.max() + 1e-14) / 255.0
        c_2 /= (c_2.max() + 1e-14) / 255.0

    c_0 = np.require(c_0, "float32", "C")
    c_1 = np.require(c_1, "float32", "C")
    c_2 = np.require(c_2, "float32", "C")

    show = np.zeros((SHOWSZ, SHOWSZ, 3), dtype="uint8")

    def render():
        """
        Rendering
        """
        rotmat = np.eye(3)

        xangle = -np.pi / 4
        rotmat = rotmat.dot(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(xangle), -np.sin(xangle)],
                    [0.0, np.sin(xangle), np.cos(xangle)],
                ]
            )
        )

        yangle = -np.pi / 4
        rotmat = rotmat.dot(
            np.array(
                [
                    [np.cos(yangle), 0.0, -np.sin(yangle)],
                    [0.0, 1.0, 0.0],
                    [np.sin(yangle), 0.0, np.cos(yangle)],
                ]
            )
        )
        rotmat *= ZOOM
        nxyz = xyz[IDX].dot(rotmat) + [SHOWSZ / 2, SHOWSZ / 2, 0]

        ixyz = nxyz.astype("int32")
        show[:] = background
        dll.render_ball(
            ct.c_int(show.shape[0]),
            ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p),
            ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p),
            c_0.ctypes.data_as(ct.c_void_p),
            c_1.ctypes.data_as(ct.c_void_p),
            c_2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius),
        )

        if magnify_blue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=0))
            if magnify_blue >= 2:
                show[:, :, 0] = np.maximum(
                    show[:, :, 0], np.roll(show[:, :, 0], -1, axis=0)
                )
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=1))
            if magnify_blue >= 2:
                show[:, :, 0] = np.maximum(
                    show[:, :, 0], np.roll(show[:, :, 0], -1, axis=1)
                )
        if showrot:
            cv2.putText(
                show,
                f"xangle {int(xangle / np.pi * 180)}",
                (30, SHOWSZ - 30),
                0,
                0.5,
                cv2.cv.CV_RGB(255, 0, 0),
            )
            cv2.putText(
                show,
                f"yangle {int(yangle / np.pi * 180)}",
                (30, SHOWSZ - 50),
                0,
                0.5,
                cv2.cv.CV_RGB(255, 0, 0),
            )
            cv2.putText(
                show,
                f"ZOOM {int(ZOOM * 100)}%",
                (30, SHOWSZ - 70),
                0,
                0.5,
                cv2.cv.CV_RGB(255, 0, 0),
            )

    render()
    return show


if __name__ == "__main__":
    np.random.seed(100)
    showpoints(np.random.randn(2500, 3))
