/** @file pose_local_parameterization.h
 *  @brief todo
 *
 *  @cite VINS-Mono
 *  @bug No known bugs.
 */

#ifndef POSE_LOCAL_PARAMETERIZATION_H
#define POSE_LOCAL_PARAMETERIZATION_H

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

/** @brief Compute the quaternion representing a small rotation.
 *  @param theta The small angle of rotation.
 */
template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar>
deltaQ(const Eigen::MatrixBase<Derived> &theta)
{
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}

class PoseLocalParameterization : public ceres::LocalParameterization
{
public:
    PoseLocalParameterization(Eigen::Matrix<double, 6, 6> _W)
        : W(_W) {}

    PoseLocalParameterization()
    {
        W.setIdentity();
    }

    /**
     * @brief Generalization of the addition operation,
                x_plus_delta = Plus(x, delta)
              with the condition that Plus(x, 0) = x.
     * @param x input
     * @param delta input
     * @param x_plus_delta output
     * @return true if ok
     */
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;

    /**
     * @brief The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
            jacobian is a row-major GlobalSize() x LocalSize() matrix.
     * @param x input
     * @param jacobian output
     * @return true if ok
     */
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };

    Eigen::Matrix<double, 6, 6> W; // eigen vectors as columns
};

#endif /* POSE_LOCAL_PARAMETERIZATION_H */