#include <iostream>
#include <chrono>

#define EIGEN_STACK_ALLOCATION_LIMIT 1000000

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#define QP_SOLVER_USE_SPARSE
#include "solvers/qp_solver.hpp"

#include <vector>
#include <numeric>
#include <random>

template <typename Scalar = double>
class Timer {
public:
    /** OS dependent */
#ifdef __APPLE__
    using clock = std::chrono::system_clock;
#else
    using clock = std::chrono::high_resolution_clock;
#endif
    using time_point = std::chrono::time_point<clock>;

    time_point _start, _stop;
    std::vector<Scalar> _samples;

    time_point get_time()
    {
        return clock::now();
    }

    void tic()
    {
        _start = get_time();
    }

    void toc()
    {
        _stop = get_time();
        Scalar t = std::chrono::duration<Scalar, std::micro>(_stop - _start).count();
        _samples.push_back(t);
    }

    void clear()
    {
        _samples.clear();
    }

    const std::vector<Scalar>& samples()
    {
        return _samples;
    }

    Scalar mean()
    {
        if (_samples.size() == 0) {
            return 0;
        }
        return std::accumulate(_samples.begin(), _samples.end(), 0.0) / _samples.size();
    }

    std::tuple<Scalar, Scalar> mean_std()
    {
        Scalar m, s;

        if (_samples.size() == 0) {
            return std::make_tuple(0.0, 0.0);
        }

        m = mean();

        std::vector<Scalar> diff(_samples.size());
        std::transform(_samples.begin(), _samples.end(), diff.begin(),
                       [m](Scalar x) {
            return x - m;
        }
                       );
        Scalar sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        s = sqrt(sq_sum / diff.size());

        return std::make_tuple(m, s);
    }

    void print()
    {
        double m, s;
        std::tie(m, s) = mean_std();
        printf("time: mean %.2f us,  std %.2f us\n", m, s);
    }
};

template <typename _Solver>
class QPBench {
public:
    using Solver = _Solver;
    using Problem = typename Solver::qp_t;
    using Scalar = typename Solver::Scalar;
    using var_t = typename Solver::var_t;

    Solver solver;

    std::tuple<int, double, double> run(const Problem& qp)
    {
        printf("problem: var %d  constr %d\n", Solver::n, Solver::m);
        var_t sol;

        solver.settings().adaptive_rho = true;
        solver.settings().alpha = 1.6;
        solver.settings().max_iter = 1000;
        solver.settings().check_termination = 25;
        solver.settings().adaptive_rho_interval = 25;

        Scalar m, m1, m2, s, s1, s2;

        Timer<Scalar> t;

        printf("setup ");
        for (int i = 0; i < 100; i++) {
            t.tic();
            solver.setup(qp);
            t.toc();
        }
        t.print();
        std::tie(m1, s1) = t.mean_std();
        t.clear();

        printf("update ");
        for (int i = 0; i < 100; i++) {
            t.tic();
            solver.setup(qp);
            t.toc();
        }
        t.print();
        t.clear();

        printf("solve ");
        for (int i = 0; i < 100; i++) {
            t.tic();
            solver.solve(qp);
            t.toc();
        }
        t.print();
        std::tie(m2, s2) = t.mean_std();

        int size = Solver::n + Solver::m;
        m = m1+m2;
        s = sqrt(s1*s1 + s2*s2);
        printf("total: mean %.2f us,  std %.2f us\n", m, s);

        printf("iter %d\n", solver.info().iter);
        // sol = solver.primal_solution();
        // std::cout << "x " << sol.transpose() << std::endl;
        return std::make_tuple(size, m, s);
    }
};

template <typename _Scalar = double>
class SimpleQP
{
public:
    using Scalar = _Scalar;
    enum {
        n=2,
        m=3
    };

    Eigen::Matrix<Scalar, n, n> P;
    Eigen::Matrix<Scalar, n, 1> q;
    Eigen::Matrix<Scalar, m, n> A;
    Eigen::Matrix<Scalar, m, 1> l, u;
    Eigen::Matrix<_Scalar, 2, 1> SOLUTION;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SimpleQP()
    {
        this->P << 4, 1,
            1, 2;
        this->q << 1, 1;
        this->A << 1, 1,
            1, 0,
            0, 1;
        this->l << 1, 0, 0;
        this->u << 1, 0.7, 0.7;

        this->SOLUTION << 0.3, 0.7;
    }
};

template <typename Scalar>
Eigen::SparseMatrix<Scalar> SpMat_gen_normal(int rows, int cols)
{
    std::default_random_engine gen;
    std::normal_distribution<Scalar> normal_dist(0.0, 1.0);
    std::uniform_int_distribution<> non_zero(0, 1);

    using T = Eigen::Triplet<Scalar>;
    std::vector<T> triplet_list;
    triplet_list.reserve(rows * cols / 2);
    for (int c = 0; c < cols; c++) {
        for (int r = 0; r < rows; r++) {
            if (non_zero(gen)) {
                Scalar val = normal_dist(gen);
                triplet_list.push_back(T(r, c, val));
            }
        }
    }
    Eigen::SparseMatrix<Scalar> mat(rows, cols);
    mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
    return mat;
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
vec_gen_normal(int len)
{
    std::default_random_engine gen;
    std::normal_distribution<Scalar> normal_dist(0.0, 1.0);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> vec(len);
    for (int i = 0; i < len; i++) {
        vec(i) = normal_dist(gen);
    }

    return vec;
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1>
vec_gen_uniform(int len)
{

    std::default_random_engine gen;
    std::uniform_real_distribution<Scalar> uniform_dist(-0.0, 1.0);

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> vec(len);
    for (int i = 0; i < len; i++) {
        vec(i) = uniform_dist(gen);
    }

    return vec;
}

template <typename Scalar>
Eigen::VectorXi count_col_nnz(const Eigen::SparseMatrix<Scalar>& mat)
{
    Eigen::VectorXi nnz(mat.cols());
    nnz.setZero();
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(mat, k); it; ++it) {
            nnz(it.col()) += 1;
        }
    }
    return nnz;
}


template <int NUM_VAR, int NUM_CONSTR, typename _Scalar = double>
struct RandomQP : qp_solver::QP<NUM_VAR, NUM_CONSTR, _Scalar>
{
    using Scalar = _Scalar;
    using SpMat = Eigen::SparseMatrix<Scalar>;

    RandomQP()
    {
        SpMat M = SpMat_gen_normal<Scalar>(NUM_VAR, NUM_VAR);
        this->P = M*M.transpose();
        this->A = SpMat_gen_normal<Scalar>(NUM_CONSTR, NUM_VAR);
#ifdef QP_SOLVER_USE_SPARSE
        this->P_col_nnz = count_col_nnz<Scalar>(this->P);
        this->A_col_nnz = count_col_nnz<Scalar>(this->A);
#endif
        this->q = vec_gen_normal<Scalar>(NUM_VAR);
        this->u = vec_gen_uniform<Scalar>(NUM_CONSTR);
        this->l = -vec_gen_uniform<Scalar>(NUM_CONSTR);

//         std::cout << "P\n" << this->P << std::endl;
//         std::cout << "q\n" << this->q.transpose() << std::endl;
//         std::cout << "A\n" << this->A << std::endl;
//         std::cout << "l\n" << this->l.transpose() << std::endl;
//         std::cout << "u\n" << this->u.transpose() << std::endl;
// #ifdef QP_SOLVER_USE_SPARSE
//         std::cout << "P nnz\n" << this->P_col_nnz.transpose() << std::endl;
//         std::cout << "A nnz\n" << this->A_col_nnz.transpose() << std::endl;
// #endif
    }
};

template <int N, int M=2*N>
std::tuple<int, double, double> RandomQP_bench()
{
    using Problem = RandomQP<N, M, double>;
#ifdef QP_SOLVER_USE_SPARSE
    using Solver = qp_solver::QPSolver<Problem, Eigen::SimplicialLDLT>;
#else
    using Solver = qp_solver::QPSolver<Problem, Eigen::LDLT>;
#endif
    QPBench<Solver> bench;
    Problem qp;
    std::tuple<int, double, double> t;
    t = bench.run(qp);
    return t;
}


int main()
{
    // using SpMat = Eigen::SparseMatrix<double>;
    // SpMat A;
    // // A = SpMat_gen_normal<double>(3, 2);
    // A = SpMat_gen_normal<double>(10, 20);

    // Eigen::Matrix<int, 20, 1> nnz = count_col_nnz<double>(A);
    // std::cout << A << std::endl;
    // std::cout << A.nonZeros() << std::endl;
    // std::cout << nnz.transpose() << std::endl;
    // std::cout << nnz.sum() << std::endl;

    // const int n = 2;
    // const int m = 10*n;
    // SpMat M = SpMat_gen_normal<double>(n, n);
    // SpMat P = M*M.transpose();
    // SpMat A = SpMat_gen_normal<double>(m, n);
    // Eigen::Matrix<double, n, 1> q = vec_gen_normal<double>(n);
    // Eigen::Matrix<double, m, 1> u = vec_gen_uniform<double>(m);
    // Eigen::Matrix<double, m, 1> l = -vec_gen_uniform<double>(m);

    // std::cout << P << std::endl;
    // std::cout << q.transpose() << std::endl;
    // std::cout << A << std::endl;
    // std::cout << l.transpose() << std::endl;
    // std::cout << u.transpose() << std::endl;

    // using Problem = SimpleQP<double>;
    // using Solver = qp_solver::QPSolver<Problem, Eigen::LDLT>;

//     const int N = 40;
//     const int M = 60; //10*N,
//     using Problem = RandomQP<N, M, double>;
// #ifdef QP_SOLVER_USE_SPARSE
//     using Solver = qp_solver::QPSolver<Problem, Eigen::SimplicialLDLT>;
// #else
//     using Solver = qp_solver::QPSolver<Problem, Eigen::LDLT>;
// #endif
//     QPBench<Solver> bench;
//     Problem qp;
//     bench.run(qp);

    std::vector<std::tuple<int, double, double>> solve_times;
    std::tuple<int, double, double> t;

    t = RandomQP_bench<1>();
    solve_times.push_back(t);
    t = RandomQP_bench<2>();
    solve_times.push_back(t);
    t = RandomQP_bench<3>();
    solve_times.push_back(t);
    t = RandomQP_bench<5>();
    solve_times.push_back(t);
    t = RandomQP_bench<7>();
    solve_times.push_back(t);
    t = RandomQP_bench<10>();
    solve_times.push_back(t);
    t = RandomQP_bench<20>();
    solve_times.push_back(t);
    t = RandomQP_bench<30>();
    solve_times.push_back(t);
    t = RandomQP_bench<50>();
    solve_times.push_back(t);
    t = RandomQP_bench<70>();
    solve_times.push_back(t);
    t = RandomQP_bench<100>();
    solve_times.push_back(t);

#ifdef QP_SOLVER_USE_SPARSE
    printf("sparse = [\n");
#else
    printf("dense = [\n");
#endif
    for (auto t : solve_times) {
        double m, s;
        int size;
        std::tie(size, m, s) = t;
        printf("[%d, %f, %f],\n", size, m, s);
    }
    printf("]\n");
}
