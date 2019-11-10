#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "ceres/rotation.h"
#include <random>
#include "tinyply.h"
#include <fstream>
struct ResidualErrorAngleAxis
{
    ResidualErrorAngleAxis(double aa[3]) {
        for (int i = 0; i < 3; i++)Wj[i] = aa[i];
    }
    template <typename T>
    bool operator()(const T* const Wi, T* residuals) const
    {
        T wj[3];
        for (int i = 0; i < 3; i++)wj[i] = T(Wj[i]);
        T Qi[4];  ceres::AngleAxisToQuaternion(Wi, Qi);
        T Qj[4];  ceres::AngleAxisToQuaternion(wj, Qj);

        T Qres[4];
        Qj[0] = -Qj[0];
        ceres::QuaternionProduct(Qj, Qi, Qres);
        ceres::QuaternionToAngleAxis(Qres, residuals);
        return true;
    }
    static ceres::CostFunction* Create(double aa[3])
    {
        return (new ceres::AutoDiffCostFunction<ResidualErrorAngleAxis, 3, 3>(new ResidualErrorAngleAxis(aa)));
    }
private:
    double Wj[3];
};

struct ResidualErrorQuaternion
{
    ResidualErrorQuaternion(double qq[4]) {
        for (int i = 0; i < 4; i++)q[i] = qq[i];
    }
    template <typename T>
    bool operator()(const T* const Wi, T* residuals) const
    {
        T Qj[4];
        for (int i = 0; i < 4; i++)Qj[i] = T(q[i]);
        Qj[0] = -Qj[0];
        ceres::QuaternionProduct(Qj, Wi, residuals);
        return true;
    }
    static ceres::CostFunction* Create(double qq[4])
    {
        return (new ceres::AutoDiffCostFunction<ResidualErrorQuaternion, 4, 4>(new ResidualErrorQuaternion(qq)));
    }
private:
    double q[4];
};

std::default_random_engine e(123);
Eigen::Vector3d RandomDir() {
    std::normal_distribution<double> g(0., 1.);
    Eigen::Vector3d v(g(e), g(e), g(e));
    return v.normalized();
}

Eigen::Matrix3d RandomRotation() {
    const Eigen::Vector3d v0 = RandomDir();
    Eigen::Vector3d v1 = RandomDir();
    v1 = v1 - v0 * (v1.dot(v0));
    v1.normalize();
    const  Eigen::Vector3d v2 = v0.cross(v1);
    Eigen::Matrix3d R;
    R.col(0) = v0;
    R.col(1) = v1;
    R.col(2) = v2;
    return R;
}

void write_rotation(const std::string& filename, const Eigen::Matrix3d& R) {
    std::vector<float> vertices;
    std::vector<uint8_t> colors;
    std::vector<uint32_t> faces;
    for (int i = 0; i < 3; i++) {
        Eigen::Vector3d c = R.col(i);
        for (int k = 0; k < 3; k++) {
            vertices.push_back(c[k]);
        }for (int k = 0; k < 3; k++) {
            vertices.push_back(0.f);
        }for (int k = 0; k < 3; k++) {
            vertices.push_back(0.f);
        }
        for (int k = 0; k < 3; k++) {
            colors.push_back(k == i ? 255 : 0);
        }
        for (int k = 0; k < 3; k++) {
            colors.push_back(k == i ? 255 : 0);
        }
        for (int k = 0; k < 3; k++) {
            colors.push_back(k == i ? 255 : 0);
        }
    }
    for (int i = 0; i < 9; i++) {
        faces.push_back(i);
    }
    std::ofstream ss(filename, std::ios::out | std::ios::binary);
    tinyply::PlyFile cube_file;

    cube_file.add_properties_to_element("vertex", { "x", "y", "z" },
        tinyply::Type::FLOAT32, 9, reinterpret_cast<uint8_t*>(vertices.data()), tinyply::Type::INVALID, 0);
    cube_file.add_properties_to_element("vertex", { "red", "green", "blue" },
        tinyply::Type::UINT8, 9, reinterpret_cast<uint8_t*>(colors.data()), tinyply::Type::INVALID, 0);


    cube_file.add_properties_to_element("face", { "vertex_indices" },
        tinyply::Type::UINT32, 3, reinterpret_cast<uint8_t*>(faces.data()), tinyply::Type::UINT8, 3);

    // Write a binary file
    cube_file.write(ss, true);
}

class MyCallBackAngleAxis : public ceres::IterationCallback
{
public:
    MyCallBackAngleAxis(double* aa) :aa(aa) {}
    double* aa;
    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
        Eigen::Matrix3d R;
        ceres::AngleAxisToRotationMatrix(aa, R.data());
        write_rotation(std::string("results/aa_") + std::to_string(summary.iteration) + ".ply", R);
        return ceres::SOLVER_CONTINUE;
    }
};

class MyCallBackQuaternion : public ceres::IterationCallback
{
public:
    MyCallBackQuaternion(double* qq) :qq(qq) {}
    double* qq;
    ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) {
        std::cout << "qq=" << qq[0] << " " << qq[1] << " " << qq[2] << " " << qq[3] << std::endl;
        Eigen::Matrix3d R;
        double aa[3];
        ceres::QuaternionToAngleAxis(qq, aa);
        ceres::AngleAxisToRotationMatrix(aa, R.data());
        write_rotation(std::string("results/qq_") + std::to_string(summary.iteration) + ".ply", R);
        return ceres::SOLVER_CONTINUE;
    }
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    const Eigen::Matrix3d R_gt = RandomRotation();
    const Eigen::Matrix3d R_init = RandomRotation();
    const float max_trust_region = 2.0;

    write_rotation("results/GT.ply", R_gt);


    if (false) {
        double aa_gt[3], aa_x[3];
        ceres::RotationMatrixToAngleAxis(R_gt.data(), aa_gt);
        ceres::RotationMatrixToAngleAxis(R_init.data(), aa_x);
        // Create residuals for each observation in the bundle adjustment problem. The
        // parameters for cameras and points are added automatically.
        ceres::Problem problem;

        ceres::CostFunction* cost_function =
            ResidualErrorAngleAxis::Create(aa_gt);
        problem.AddResidualBlock(cost_function, NULL /* squared loss */, aa_x);

        MyCallBackAngleAxis call(aa_x);

        ceres::Solver::Options options;
        options.update_state_every_iteration = true;
        options.callbacks.push_back(&call);
        options.max_trust_region_radius = max_trust_region;
        options.initial_trust_region_radius = max_trust_region;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.logging_type = ceres::PER_MINIMIZER_ITERATION;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";
    }
    double qq_gt[4], qq_x[4];
    {
        ceres::RotationMatrixToQuaternion(R_gt.data(), qq_gt);
        ceres::RotationMatrixToQuaternion(R_init.data(), qq_x);
        std::cout << "qq_gt=" << qq_gt[0] << " " << qq_gt[1] << " " << qq_gt[2] << " " << qq_gt[3] << std::endl;

        ceres::Problem problem;

        ceres::CostFunction* cost_function =
            ResidualErrorQuaternion::Create(qq_gt);
        problem.AddResidualBlock(cost_function, NULL /* squared loss */, qq_x);

        MyCallBackQuaternion call(qq_x);

        ceres::Solver::Options options;
        options.update_state_every_iteration = true;
        options.callbacks.push_back(&call);
        options.max_trust_region_radius = max_trust_region;
        options.initial_trust_region_radius = max_trust_region;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = true;
        options.logging_type = ceres::PER_MINIMIZER_ITERATION;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";
    }


    Eigen::Matrix3d Rqq;
    double aaq[3];
    ceres::QuaternionToAngleAxis(qq_x, aaq);
    ceres::AngleAxisToRotationMatrix(aaq, Rqq.data());
    write_rotation(std::string("results/qq_final.ply"), Rqq);

    return 0;
}