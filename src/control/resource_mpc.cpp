#include <iostream>
#include <fstream>
#include <vector>

#include "polynomials/ebyshev.hpp"
#include "control/double_integrator_model.hpp"
#include "control/nmpc.hpp"
#include "solvers/sqp.hpp"


template <typename Solver>
void callback(void *solver_p)
{
    Solver& s = *static_cast<Solver*>(solver_p);
    std::cout << s._x.transpose() << std::endl;
}

using Problem = polympc::OCProblem<System<double>, Lagrange<double>, Mayer<double>>;
using Approximation = Chebyshev<3, GAUSS_LOBATTO, double>; // POLY_ORDER = 3

using controller_t = polympc::nmpc<Problem, Approximation, sqp::SQP>;
using var_t = controller_t::var_t;
using dual_t = controller_t::dual_t;
using State = controller_t::State;
using Control = controller_t::Control;
using Parameters = controller_t::Parameters;

enum
{
    NX = State::RowsAtCompileTime,
    NU = Control::RowsAtCompileTime,
    NP = Parameters::RowsAtCompileTime,
    VARX_SIZE = controller_t::VARX_SIZE,
    VARU_SIZE = controller_t::VARU_SIZE,
    VARP_SIZE = controller_t::VARP_SIZE,
    VAR_SIZE = var_t::RowsAtCompileTime,
};

void print_info(void)
{
    std::cout << "NX = " << NX << "\n";
    std::cout << "NU = " << NU << "\n";
    std::cout << "NP = " << NP << "\n";
    std::cout << "VARX_SIZE = " << VARX_SIZE << "\n";
    std::cout << "VARU_SIZE = " << VARU_SIZE << "\n";
    std::cout << "VARP_SIZE = " << VARP_SIZE << "\n";
    std::cout << "VAR_SIZE = " <<  VAR_SIZE << "\n";

    std::cout << "controller_t size: " << sizeof(controller_t) << "\n";
    std::cout << "controller_t::cost_colloc_t size: " << sizeof(controller_t::cost_colloc_t) << "\n";
    std::cout << "controller_t::ode_colloc_t size: " << sizeof(controller_t::ode_colloc_t) << "\n";
    std::cout << "controller_t::sqp_t size: " << sizeof(controller_t::SolverImpl) << "\n";
    std::cout << "controller_t::sqp_t::qp_t size: " << sizeof(controller_t::SolverImpl::qp_t) << "\n";
    std::cout << "controller_t::sqp_t::qp_solver_t size: " << sizeof(controller_t::SolverImpl::qp_solver_t) << "\n";
}

void print_duals(const dual_t& y)
{
    Eigen::IOFormat fmt(3, 0, ", ", ",", "[", "]");
    std::cout << "duals" << std::endl;
    std::cout << "ode   " << y.template segment<VARX_SIZE>(0).transpose().format(fmt) << std::endl;
    std::cout << "x     " << y.template segment<VARX_SIZE-NX>(VARX_SIZE).transpose().format(fmt) << std::endl;
    std::cout << "x0    " << y.template segment<NX>(2*VARX_SIZE-NX).transpose().format(fmt) << std::endl;
    std::cout << "u     " << y.template segment<VARU_SIZE>(2*VARX_SIZE).transpose().format(fmt) << std::endl;
}

void print_sol(const var_t& sol)
{
    Eigen::IOFormat fmt(4, 0, ", ", ",", "[", "]");
    std::cout << "xyt" << std::endl;
    for (int i = 0; i < VARX_SIZE/NX; i++) {
       std::cout << sol.segment<NX>(NX*i).transpose().format(fmt) << ",\n";
    }
    std::cout << "u" << std::endl;
    for (int i = 0; i < VARU_SIZE/NU; i++) {
       std::cout << sol.segment<NU>(VARX_SIZE+NU*i).transpose().format(fmt) << ",\n";
    }
}

template <typename Var>
void save_csv(const char *name, const std::vector<Var>& vec)
{
    std::ofstream out(name);
    Eigen::IOFormat fmt(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",", "", "");
    for (const Var& x : vec) {
        out << x.transpose().format(fmt) << "\n";
    }
}

template<typename Derived>
inline bool is_nan(const Eigen::MatrixBase<Derived>& x)
{
    return !((x.array() == x.array())).all();
}



int main(void)
{
    controller_t robot_controller;
    // robot_controller.m_solver.settings().iteration_callback = callback<controller_t::sqp_t>;

    State x = {0, 0, 0.5};
    State dx;
    Control u;
    Parameters p(1.0); //ref point

    // bounds
    controller_t::State xu, xl;
    xu << -100, -5, 0.0;
    xl <<  100,  5, 1.0;
    controller_t::Control uu, ul;
    uu << -10,  0.5, 0.25;
    ul <<  10,  2.0, 0.5;

    robot_controller.setStateBounds(xl, xu);
    robot_controller.setControlBounds(ul, uu);
    robot_controller.setParameters(p);
    robot_controller.disableWarmStart();

    print_info();
    std::cout << "x0: " << x.transpose() << std::endl;

    robot_controller.computeControl(x);
    robot_controller.getOptimalControl(u);

    var_t sol;
    robot_controller.getSolution(sol);

    std::cout << "Solution: " << sol << "\n";

    std::cout << "iter " << robot_controller.m_solver.info().iter << "  ";
    std::cout << "qp " << robot_controller.m_solver.info().qp_solver_iter << "  ";
    std::cout << "x " << x.transpose() << "    ";
    std::cout << "u " << u.transpose() << std::endl;

    return 0;
}
