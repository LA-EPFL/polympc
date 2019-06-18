#include <stdio.h>
#include <iostream>

#include "timer.hpp"
#include "solvers/sqp.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

template <typename _Derived, typename _Scalar, int _VAR_SIZE, int _NUM_EQ=0, int _NUM_INEQ=0>
struct ProblemBase {
    enum {
        VAR_SIZE = _VAR_SIZE,
        NUM_EQ = _NUM_EQ,
        NUM_INEQ = _NUM_INEQ,
    };

    using Scalar = double;
    using var_t = Eigen::Matrix<Scalar, VAR_SIZE, 1>;
    using grad_t = Eigen::Matrix<Scalar, VAR_SIZE, 1>;
    using hessian_t = Eigen::Matrix<Scalar, VAR_SIZE, VAR_SIZE>;
    using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    using b_eq_t = Eigen::Matrix<Scalar, NUM_EQ, 1>;
    using A_eq_t = Eigen::Matrix<Scalar, NUM_EQ, VAR_SIZE>;
    using b_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, 1>;
    using A_ineq_t = Eigen::Matrix<Scalar, NUM_INEQ, VAR_SIZE>;
    using box_t = var_t;

    using ADScalar = Eigen::AutoDiffScalar<grad_t>;
    using ad_var_t = Eigen::Matrix<ADScalar, VAR_SIZE, 1>;
    using ad_eq_t = Eigen::Matrix<ADScalar, NUM_EQ, 1>;
    using ad_ineq_t = Eigen::Matrix<ADScalar, NUM_INEQ, 1>;

    template <typename vec>
    void AD_seed(vec &x)
    {
        for (int i=0; i<x.rows(); i++) {
            x[i].derivatives().coeffRef(i) = 1;
        }
    }

    void cost_linearized(const var_t& x, grad_t &grad, Scalar &cst)
    {
        ad_var_t _x = x;
        ADScalar _cst;
        AD_seed(_x);
        /* Static polymorphism using CRTP */
        static_cast<_Derived*>(this)->cost(_x, _cst);
        cst = _cst.value();
        grad = _cst.derivatives();
    }

    void constraint_linearized(const var_t& x, A_eq_t& A_eq, b_eq_t& b_eq, A_ineq_t& A_ineq, b_ineq_t& b_ineq, box_t& lbx, box_t& ubx)
    {
        ad_eq_t ad_eq;
        ad_ineq_t ad_ineq;

        ad_var_t _x = x;
        AD_seed(_x);
        static_cast<_Derived*>(this)->constraint(_x, ad_eq, ad_ineq, lbx, ubx);

        for (int i = 0; i < ad_eq.rows(); i++) {
            b_eq[i] = ad_eq[i].value();
            Eigen::Ref<MatX> deriv = ad_eq[i].derivatives().transpose();
            A_eq.row(i) = deriv;
        }

        for (int i = 0; i < ad_ineq.rows(); i++) {
            b_ineq[i] = ad_ineq[i].value();
            Eigen::Ref<MatX> deriv = ad_ineq[i].derivatives().transpose();
            A_ineq.row(i) = deriv;
        }
    }
};

struct Rosenbrock : public ProblemBase<Rosenbrock,
                                       double,
                                       /* Nx    */2,
                                       /* Neq   */0,
                                       /* Nineq */0>  {
    const Scalar a = 1;
    const Scalar b = 100;
    Eigen::Vector2d SOLUTION = {1.0, 1.0};

    template <typename DerivedA, typename DerivedB>
    void cost(const DerivedA& x, DerivedB &cst)
    {
        // (a-x)^2 + b*(y-x^2)^2
        cst = pow(a - x(0), 2) + b * pow(x(1) - pow(x(0), 2), 2);
    }

    template <typename A, typename B, typename C>
    void constraint(const A& x, B& eq, C& ineq, box_t& lbx, box_t& ubx)
    {
        // unconstrained
        const Scalar infinity = std::numeric_limits<Scalar>::infinity();
        lbx << -infinity, -infinity;
        ubx << infinity, infinity;
    }
};

struct Rosenbrock10 : public ProblemBase<Rosenbrock10,
                                       double,
                                       /* Nx    */10,
                                       /* Neq   */0,
                                       /* Nineq */0>  {
    const int N = 10;
    using Scalar = double;
    const Scalar a = 1;
    const Scalar b = 100;

    template <typename DerivedA, typename DerivedB>
    void cost(const DerivedA& x, DerivedB &cst)
    {
        cst = 0;
        for (int i = 0; i < N-1; i++) {
            cst += pow(a - x[i], 2) + b * pow(x[i+1] - pow(x[i], 2), 2);
        }
    }

    template <typename A, typename B, typename C>
    void constraint(const A& x, B& eq, C& ineq, box_t& lbx, box_t& ubx)
    {
        // unconstrained
        const Scalar infinity = std::numeric_limits<Scalar>::infinity();
        lbx.setConstant(-infinity);
        ubx.setConstant(infinity);
    }
};

struct Rosenbrock50 : public ProblemBase<Rosenbrock50,
                                       double,
                                       /* Nx    */50,
                                       /* Neq   */0,
                                       /* Nineq */0>  {
    const int N = 50;
    using Scalar = double;
    const Scalar a = 1;
    const Scalar b = 100;

    template <typename DerivedA, typename DerivedB>
    void cost(const DerivedA& x, DerivedB &cst)
    {
        cst = 0;
        for (int i = 0; i < N-1; i++) {
            cst += pow(a - x[i], 2) + b * pow(x[i+1] - pow(x[i], 2), 2);
        }
    }

    template <typename A, typename B, typename C>
    void constraint(const A& x, B& eq, C& ineq, box_t& lbx, box_t& ubx)
    {
        // unconstrained
        const Scalar infinity = std::numeric_limits<Scalar>::infinity();
        lbx.setConstant(-infinity);
        ubx.setConstant(infinity);
    }
};

template <typename Solver>
void callback(void *solver_p)
{
    Solver& s = *static_cast<Solver*>(solver_p);

    Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", ",", "[", "],");
    std::cout << s._x.transpose().format(fmt) << std::endl;
}

template <typename _Problem>
void test_problem()
{
    using Problem = _Problem;
    using Solver = sqp::SQP<Problem>;
    using var_t = typename Solver::var_t;
    using dual_t = typename Solver::dual_t;
    Problem problem;
    Solver solver;
    Timer t;

    std::cout << "N " << var_t::RowsAtCompileTime << std::endl;

    var_t x;
    var_t x0;
    dual_t y0;
    x0.setZero();
    y0.setZero();

    solver.settings().max_iter = 100;
    // solver.settings().iteration_callback = callback<Solver>;

    for (int i = 0; i < 100; i++) {
        t.tic();
        solver.solve(problem, x0, y0);
        t.toc();
    }
    t.print();
    t.clear();

    x = solver.primal_solution();

    std::cout << "iter " << solver.info().iter << std::endl;
    std::cout << "qp_solver_iter " << solver.info().qp_solver_iter << std::endl;
    // std::cout << "Solution " << x.transpose() << std::endl;
}

int main()
{
    test_problem<Rosenbrock>();
    test_problem<Rosenbrock10>();
    test_problem<Rosenbrock50>();
}
