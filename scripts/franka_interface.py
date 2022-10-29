import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm

if __name__ == " main ":
    fa = FrankaArm()
    fa.reset_joints()
    # ee pose
    T_ee_world = fa.get_pose()
    print("Translation: {}".format(T_ee_world.translation))
    print("Rotationn: {}".format(T_ee_world.quaternion))
    # joint angles
    joints = fa.get_joints()
    print("Joints: {}".format(joints))
    # Gripper width
    gripper_width = fa.get_gripper_width()
    print("Gripper width: {}".format(gripper_width))
    # Endeffector forces
    force_torque = fa.get_ee_force_torque()
    print("Forces and Torques: {}".format(force_torque))

    # Arm
    fa.goto_joints([0.0, -0.7, 0.0, -2.15, 0.0, 1.57, 0.7])
    # Gripper
    fa.close_gripper()
    fa.open_gripper()
    fa.goto_gripper(0.03)

    des_pose = RigidTransform(
        rotation=np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        ),
        translation=np.array([0.3, -0.2, 0.4]),
        from_frame="franka_tool",
        to_frame="world",
    )
    fa.goto_pose(des_pose, use_impedance=False)

    fa.goto_pose(
        des_pose,
        duration=10.0,
        use_impedance=False,
        cartesian_impedances=[3000, 3000, 100, 300, 300, 300],
    )
