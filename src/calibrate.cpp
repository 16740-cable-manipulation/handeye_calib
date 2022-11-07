/**
 * @file calibrate.cpp
 * @author Tina Tian (yutian)
 * @brief Upon receiving computation request, compute T_e_c and send back
 * @date 10/29/2022
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "calibrate.h"

#include "cost_function.h"

Calibrator::Calibrator(ros::NodeHandle *nodehandle) : nh(*nodehandle)
{
    ROS_INFO("Initializing Calibrator...");
    calib_server = nh.advertiseService("/calibrate",
                                       &Calibrator::calibCallback,
                                       this);
}

bool Calibrator::
    calibCallback(handeye_calib::HandEyeCalibrationRequest &request,
                  handeye_calib::HandEyeCalibrationResponse &response)
{
    // request.points should contain request.num * request.k points
    // request.T_b_e should contain request.k poses
    if (request.points.size() != request.num * request.k)
    {
        ROS_ERROR("[calibrator] Incorrect point number");
        return false;
    }
    double pose[7] = {0.045, -0.04, 0.05, 0, 0, 0.7071068, 0.7071068};
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();

    problem.AddParameterBlock(pose, 7, local_parameterization);
    for (int p = 0; p < request.num; p++) // the point's position on the board
    {
        for (int i = 0; i < request.k - 1; i++) // pose index
        {
            // get p_i, T_i
            Eigen::Vector3d pc_i = PointMsg2Eigen(request.points[i * request.num + p]);
            Eigen::Quaterniond q_i;
            Eigen::Vector3d t_i;
            PoseMsg2qt(request.T_b_e[i], q_i, t_i);
            for (int j = i + 1; j < request.k; j++) // pose index
            {
                // get p_j, T_j
                Eigen::Vector3d pc_j = PointMsg2Eigen(request.points[j * request.num + p]);
                Eigen::Quaterniond q_j;
                Eigen::Vector3d t_j;
                PoseMsg2qt(request.T_b_e[j], q_j, t_j);
                auto cost_function =
                    HandEyeCostFunction::Create(pc_i,
                                                pc_j,
                                                q_i, t_i,
                                                q_j, t_j);
                problem.AddResidualBlock(cost_function, NULL, pose);
            }
        }
    }
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    // options.use_nonmonotonic_steps = true;
    // options.max_solver_time_in_seconds = 1;
    // options.max_num_iterations = 100;
    // options.use_inner_iterations = true;
    options.minimizer_progress_to_stdout = true;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;
    cout << "Iterations : " << static_cast<int>(summary.iterations.size()) << endl;
    geometry_msgs::Pose pose_out;
    pose_out.position.x = pose[0];
    pose_out.position.y = pose[1];
    pose_out.position.z = pose[2];
    pose_out.orientation.x = pose[3];
    pose_out.orientation.y = pose[4];
    pose_out.orientation.z = pose[5];
    pose_out.orientation.w = pose[6];
    response.T_e_c = pose_out;
    return true;
}

Eigen::Vector3d PointMsg2Eigen(geometry_msgs::Point &pt)
{
    Eigen::Vector3d res;
    res << pt.x, pt.y, pt.z;
    return res;
}

void PoseMsg2qt(const geometry_msgs::Pose &pose_msg,
                Eigen::Quaterniond &q, Eigen::Vector3d &t)
{
    t[0] = pose_msg.position.x;
    t[1] = pose_msg.position.y;
    t[2] = pose_msg.position.z;
    q.w() = pose_msg.orientation.w;
    q.x() = pose_msg.orientation.x;
    q.y() = pose_msg.orientation.y;
    q.z() = pose_msg.orientation.z;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "calibrator");
    ros::NodeHandle nh;

    ROS_INFO("Instantiating an object of type Calibrator");
    Calibrator calibrator(&nh);

    ROS_INFO("main: Going into spin; let the callbacks do all the work");
    ros::spin();
    return 0;
}