"""handeye_calib_node

1. Read in params
2. Command arm to move
3. Take an image
4. Extract checkerboard corners and compute PnP
5. Concatenate corners and poses and send calibrate request to Calibrator
6. Dump the result
"""

import imp
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
import pyrealsense2 as rs
from geometry_msgs.msg import Point, Pose
import rospy

from handeye_calib.srv import HandEyeCalibration

import sys, os
import numpy as np
import cv2
import yaml


def toPointMsg(pt):
    return Point(pt[0], pt[1], pt[2])


class HandEyeCalibrationNode:
    def __init__(self, base_folder, checker_dim, checker_size):
        # termination criteria
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        self.criteria_cal = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            1e-5,
        )

        # Arrays to store object points and image points from all the images.
        self.pc = []  # 3d object points in camera frame, length n*k
        self.T_b_e = []  # poses from ee to robot body frame, length k

        self.num_valid_frames = 0

        self.cal_path = base_folder
        self.checker_row = checker_dim[0]
        self.check_col = checker_dim[1]
        self.checker_dim = checker_dim
        self.checker_size = checker_size  # in mm

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((self.checker_row * self.check_col, 3), np.float32)
        self.objp[:, :2] = (
            np.mgrid[0 : self.checker_row, 0 : self.check_col].T.reshape(-1, 2)
            * self.checker_size
        )

        self.ee_poses = []  # a list of poses the ee should go to
        self.T_e_c = None

    def process_image(self, img, pose_id, T_b_e):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]  # width, height

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, self.checker_dim, None)
        vis = img
        # If found, add object points, image points (after refining them)
        if ret is True:
            print("pose #", pose_id, " valid")
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self.criteria
            )
            # Draw and display the corners
            vis = cv2.drawChessboardCorners(img, self.checker_dim, corners, ret)
            # cv2.imshow(images_left[i], img_l)
            # pnp
            success, rvec, tvec = cv2.solvePnP(
                self.objp, corners, self.K, self.D, flags=0
            )
            if success is True:
                # transform objp to camera frame
                T_w_c = 0
                pc = (T_w_c.T @ self.objp.T).T
                for i in pc.shape[0]:
                    self.pc.append(toPointMsg(pc[i, :]))
                self.T_b_e.append(T_b_e)  # pose msg
                self.num_valid_frames += 1
        else:
            print("pose #", pose_id, " invalid")
        # visualize
        cv2.imshow("image", vis)
        cv2.waitKey(300)

    def request_calibration(self):
        rospy.wait_for_service("/calibrate")
        try:
            calib_client = rospy.ServiceProxy("/calibrate", HandEyeCalibration)
            resp = calib_client(
                self.pc,
                self.T_b_e,
                self.check_col * self.checker_row,
                self.num_valid_frames,
            )
            return resp.T_e_c
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def write_result(self):
        if self.T_e_c is not None:
            rot = [
                self.T_e_c.orientation.x,
                self.T_e_c.orientation.y,
                self.T_e_c.orientation.z,
                self.T_e_c.orientation.w,
            ]
            trans = [
                self.T_e_c.position.x,
                self.T_e_c.position.y,
                self.T_e_c.position.z,
            ]
            calib_result = {"rot_e_c": rot, "trans_e_c": trans}
            with open("test.yaml", "w") as f:
                yaml.dump(calib_result, f)


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print(
            "Usage: python stereo_calibration img_base_folder trial_folder"
            " checker_row checker_col checker_size_m"
        )
        exit(0)

    base_folder = sys.argv[1]
    trial_folder = sys.argv[2]
    checker_row = int(sys.argv[3])
    checker_col = int(sys.argv[4])
    checker_size = float(sys.argv[5])

    cal_data = HandEyeCalibrationNode(
        base_folder, (checker_row, checker_col), checker_size
    )
