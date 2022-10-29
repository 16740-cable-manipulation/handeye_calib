/**
 * @file cost_function.h
 * @author Tina Tian (yutian)
 * @brief
 * @date 10/29/2022
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef HANDEYE_COST_FUNCTION_H
#define HANDEYE_COST_FUNCTION_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

#include "pose_local_parameterization.h"

struct HandEyeCostFunction
{
    const Eigen::Vector3d pc_1, pc_2;
    Eigen::Quaterniond q_b_e_1, q_b_e_2;
    Eigen::Vector3d t_b_e_1, t_b_e_2;

    HandEyeCostFunction(const Eigen::Vector3d &_pc_1,
                        const Eigen::Vector3d &_pc_2,
                        Eigen::Quaterniond &_q_1,
                        Eigen::Vector3d &_t_1,
                        Eigen::Quaterniond &_q_2,
                        Eigen::Vector3d &_t_2)
        : pc_1(_pc_1), pc_2(_pc_2),
          q_b_e_1(_q_1), q_b_e_2(_q_2),
          t_b_e_1(_t_1), t_b_e_2(_t_2) {}

    template <typename T>
    bool operator()(const T *const tf_e_c, T *residuals) const
    {
        Eigen::Matrix<T, 3, 1> p1;
        p1 << T(pc_1[0]), T(pc_1[1]), T(pc_1[2]);
        Eigen::Matrix<T, 3, 1> p2;
        p2 << T(pc_2[0]), T(pc_2[1]), T(pc_2[2]);

        Eigen::Matrix<T, 3, 1> t1;
        t1 << T(t_b_e_1[0]), T(t_b_e_1[1]), T(t_b_e_1[2]);
        Eigen::Matrix<T, 3, 1> t2;
        t2 << T(t_b_e_2[0]), T(t_b_e_2[1]), T(t_b_e_2[2]);

        Eigen::Quaternion<T> q1(T(q_b_e_1.w()), T(q_b_e_1.x()), T(q_b_e_1.y()), T(q_b_e_1.z()));
        Eigen::Quaternion<T> q2(T(q_b_e_2.w()), T(q_b_e_2.x()), T(q_b_e_2.y()), T(q_b_e_2.z()));

        Eigen::Matrix<T, 3, 1> t_e_c =
            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(tf_e_c);
        Eigen::Quaternion<T> q_e_c =
            Eigen::Map<const Eigen::Quaternion<T>>(tf_e_c + 3);

        // T_b_e(i) * T_e_c * p_c(i) = T_b_e(j) * T_e_c * p_c(j)
        Eigen::Matrix<T, 3, 1> p1_b = q_e_c * p1 + t_e_c; // p1_e
        p1_b = q1 * p1_b + t1;

        Eigen::Matrix<T, 3, 1> p2_b = q_e_c * p2 + t_e_c; // p2_e
        p2_b = q2 * p2_b + t2;

        residuals[0] = (p1_b - p2_b).dot(p1_b - p2_b);
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d &_pc_1,
                                       const Eigen::Vector3d &_pc_2,
                                       Eigen::Quaterniond &_q_1,
                                       Eigen::Vector3d &_t_1,
                                       Eigen::Quaterniond &_q_2,
                                       Eigen::Vector3d &_t_2)
    {
        return (new ceres::AutoDiffCostFunction<HandEyeCostFunction, 1, 7>(
            new HandEyeCostFunction(_pc_1, _pc_2, _q_1, _t_1, _q_2, _t_2)));
    }
};

#endif