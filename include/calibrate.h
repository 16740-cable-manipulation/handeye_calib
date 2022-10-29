/**
 * @file calibrate.h
 * @author Tina Tian (yutian)
 * @brief
 * @date 10/29/2022
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef HANDEYE_CALIBRATE_H
#define HANDEYE_CALIBRATE_H

#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <handeye_calib/HandEyeCalibration.h>

using namespace std;

class Calibrator
{
public:
    Calibrator(ros::NodeHandle *nodehandle);

    ros::NodeHandle nh;

    // add service server
    ros::ServiceServer calib_server;

    bool calibCallback(handeye_calib::HandEyeCalibrationRequest &request,
                       handeye_calib::HandEyeCalibrationResponse &response);
};

void PoseMsg2qt(const geometry_msgs::Pose &pose_msg,
                Eigen::Quaterniond &q, Eigen::Vector3d &t);

Eigen::Vector3d PointMsg2Eigen(geometry_msgs::Point &pt);

#endif