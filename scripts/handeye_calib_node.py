#!/home/student/Prog/iamEnv/bin/python
"""handeye_calib_node

1. Read in params
2. Command arm to move
3. Take an image
4. Extract checkerboard corners and compute PnP
5. Concatenate corners and poses and send calibrate request to Calibrator
6. Dump the result
"""

import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
import pyrealsense2 as rs
from geometry_msgs.msg import Point, Pose
import rospy
from scipy.spatial.transform import Rotation as R

from handeye_calib.srv import HandEyeCalibration
import time

import sys, os
import numpy as np
import cv2
from cv_bridge import CvBridge
import yaml

R_e_c_guess = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
t_e_c_guess = np.array([0.045, -0.04, 0.05])
t_c_e_guess = -t_e_c_guess


def toPointMsg(pt):
    return Point(pt[0], pt[1], pt[2])


def pose2PoseMsg(pose):
    """``pose``: {"R": quat (xyzw), "t": trans}"""
    res = Pose()
    res.position.x = pose["t"][0]
    res.position.y = pose["t"][1]
    res.position.z = pose["t"][2]
    # rot = R.from_matrix(pose["R"])
    # q = rot.as_quat()
    q = pose["R"]
    res.orientation.x = q[0]
    res.orientation.y = q[1]
    res.orientation.z = q[2]
    res.orientation.w = q[3]
    return res


class Realsense:
    def __init__(self):
        self.seq = 0
        self.time_offset = None
        self.fnumber_captured = False
        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.depth)
        self.cfg.enable_stream(rs.stream.color)

        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.pc = rs.pointcloud()
        self.frame_buffer = []
        self.use_trigger = False
        self.bridge = CvBridge()
        self.K = None
        self.D = None

        self.setSyncMode(_use_trigger=self.use_trigger)
        self.start()

    def start(self):
        profile = self.pipeline.start(self.cfg)
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        print(intr)
        self.K = np.array(
            [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1]]
        )
        self.D = np.array(intr.coeffs)
        print("Realsense ready!")

    def setSyncMode(self, _use_trigger=True):
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        r = self.cfg.resolve(pipeline_wrapper)
        device = r.get_device()
        sensor = device.query_sensors()[0]
        # set sensor sync mode
        self.use_trigger = _use_trigger
        if self.use_trigger:
            sensor.set_option(rs.option.inter_cam_sync_mode, 1)
        else:
            sensor.set_option(rs.option.inter_cam_sync_mode, 0)

    def getFrameSet(self):
        """Get rs frames and return color image"""
        frameset = self.pipeline.wait_for_frames()
        rgb = self.getPointCloudMsg(frameset)
        return rgb

    def getPointCloudMsg(self, frameset):
        """Construct PointCloud2 msg given rs frameset and ros timestamp"""
        if self.use_trigger:
            fnumber = frameset.get_frame_number()
            # make sure we only publish one frame corresponding to this fnumber
            # or if fnumber != seq+1, the frame might be corrupted
            if fnumber == self.seq or fnumber != self.seq + 1:
                return
        # Align the depth frame to color frame
        aligned_frames = self.align.process(frameset)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        return color_image

    def close(self):
        print("Closing Realsense")
        self.pipeline.stop()


class HandEyeCalibrationNode:
    def __init__(self, checker_center, checker_dim, checker_size, use_rs=False):
        self.use_rs = use_rs
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

        self.ee_poses = self.generate_poses(checker_center)[::2]
        self.T_e_c = None

        if use_rs is True:
            self.realsense = Realsense()

        self.fa = FrankaArm()
        self.fa.reset_joints()
        self.fa.open_gripper()
        time.sleep(4)
        T_ee_world = self.fa.get_pose()
        print("Translation: {}".format(T_ee_world.translation))
        print("Rotationn: {}".format(T_ee_world.quaternion))

    def compute_rotation(self, checker_center, trans):
        vz = checker_center - trans
        vz = vz / np.linalg.norm(vz)
        vx = np.array([1.0, 0.0, 0.0])
        vy = np.cross(vz, vx)
        vy = vy / np.linalg.norm(vy)
        Rot = np.vstack((vx, vy, vz)).T
        rot = R.from_matrix(Rot)
        return rot.as_matrix()

    def generate_poses(self, checker_center):
        # a list of poses the ee should go to
        dx1 = 0.10
        dx2 = 0.10
        dy1 = 0.2
        dy2 = 0.2
        z0 = 0.25
        dz = 0.35
        nx = 5
        ny = 5
        nz = 5
        cx, cy, cz = checker_center
        x_ = np.linspace(cx - dx1, cx + dx2, nx)
        y_ = np.linspace(cy - dy1, cy + dy2, ny)
        z_ = np.linspace(z0, z0 + dz, nz)

        x, y, z = np.meshgrid(x_, y_, z_, indexing="ij")
        # at each (x,y,z) point, compute the rotation
        translations = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        rotations = [
            self.compute_rotation(np.array([cx, cy, cz]), trans)
            for trans in translations
        ]
        translations = [
            np.dot(rotations[i], t_c_e_guess.reshape((-1, 1)))
            + translations[i].reshape((-1, 1))
            for i in range(len(rotations))
        ]
        return [
            {"R": rotations[i], "t": translations[i]}
            for i in range(len(translations))
        ]

    def command_pose(self, _rotation, _translation):
        """
        Command ee to go to a pose and take an picture.

        ``_rotation``: np array, rotation matrix, size 3x3
        ``_translation``: np array, translation vector size 3
        """
        des_pose = RigidTransform(
            rotation=_rotation,
            translation=_translation,
            from_frame="franka_tool",
            to_frame="world",
        )
        self.fa.goto_pose(des_pose, use_impedance=False)
        time.sleep(4)
        img = None
        T_ee_world = self.fa.get_pose()
        trans_actual = T_ee_world.translation
        rot_actual = T_ee_world.quaternion  # xyzw
        pose_actual = {"R": rot_actual, "t": trans_actual}
        if self.use_rs:
            img = self.realsense.getFrameSet()
        return img, pose_actual

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
                self.objp, corners, self.realsense.K, self.realsense.D, flags=0
            )
            if success is True:
                # transform objp to camera frame
                T_c_w = np.eye(4)
                rmat, _ = cv2.Rodrigues(rvec)
                T_c_w[:3, :3] = rmat
                T_c_w[:3, 3] = tvec.flatten()
                objp_homo = np.hstack(
                    (self.objp, np.ones((self.objp.shape[0], 1)))
                )
                pc = np.matmul(T_c_w, objp_homo.T).T
                pc = pc[:, :3]
                for i in range(pc.shape[0]):
                    self.pc.append(toPointMsg(pc[i, :]))
                self.T_b_e.append(T_b_e)  # pose msg
                self.num_valid_frames += 1
        else:
            print("pose #", pose_id, " invalid")
        # visualize
        cv2.imshow("image", vis)
        cv2.waitKey(2000)

    def request_calibration(self):
        if len(self.T_b_e) < 2:
            RuntimeError("Not enough poses!")
            quit(1)
        rospy.wait_for_service("/calibrate")
        try:
            print("Waiting for service")
            calib_client = rospy.ServiceProxy("/calibrate", HandEyeCalibration)
            print("Got service.")
            resp = calib_client(
                self.pc,
                self.T_b_e,
                self.check_col * self.checker_row,
                self.num_valid_frames,
            )
            self.T_e_c = resp.T_e_c
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
            print("calib result: ")
            print(calib_result)
            with open("test.yaml", "w") as f:
                yaml.dump(calib_result, f)

    def run(self):
        for pose_id, pose in enumerate(self.ee_poses):
            print("pose #", pose_id, ":")
            print(pose)
            img, pose_actual = self.command_pose(pose["R"], pose["t"])
            print(pose_actual)
            if img is not None:
                T_b_e = pose2PoseMsg(pose_actual)
                self.process_image(img, pose_id, T_b_e)
        self.realsense.close()
        self.request_calibration()
        self.write_result()

    def run_pose_only(self):
        for pose_id, pose in enumerate(self.ee_poses):
            print("pose #", pose_id, ":")
            print(pose)
            img, pose_actual = self.command_pose(pose["R"], pose["t"])
            if img is not None:
                T_b_e = pose2PoseMsg(pose_actual)
                self.process_image(img, pose_id, T_b_e)


if __name__ == "__main__":
    # rospy.init_node("handeye_calib_node")
    """
    if len(sys.argv) < 7:
        print(
            "Usage: python3 handeye_calib_node checker_x checker_y checker_z "
            "checker_row checker_col checker_size_m"
        )
        exit(0)

    checker_x = float(sys.argv[1])
    checker_y = float(sys.argv[2])
    checker_z = float(sys.argv[3])
    checker_row = int(sys.argv[4])
    checker_col = int(sys.argv[5])
    checker_size = float(sys.argv[6])
    """
    checker_x = 0.4
    checker_y = 0.0
    checker_z = 0.0
    checker_row = 7
    checker_col = 10
    checker_size = 0.02

    cal_data = HandEyeCalibrationNode(
        (checker_x, checker_y, checker_z),
        (checker_row, checker_col),
        checker_size,
        use_rs=True,
    )
    # quit()
    cal_data.run()
