#include "nmpc.hpp"

// define the system dynamics
class System
{
public:
    System();
    ~System(){}

    casadi::Function getDynamics(){return NumDynamics;}
    casadi::Function getOutputMapping(){return OutputMap;}
private:
    casadi::SX state;
    casadi::SX control;
    casadi::SX Dynamics;

    casadi::Function NumDynamics;
    casadi::Function OutputMap;
};

System::System()
{
    casadi::SX u = casadi::SX::sym("u");
    casadi::SX D = casadi::SX::sym("D");

    casadi::SX x = casadi::SX::sym("x", 2);

    state = x;
    control = casadi::SX::vertcat({u,D});

    Dynamics = casadi::SX::vertcat({D * u, D * x(0)});
    NumDynamics = casadi::Function("Dynamics", {state, control}, {Dynamics});

    /** define output mapping */
    OutputMap = casadi::Function("Map",{state}, {x(1)});
}

using namespace casadi;

int main(void)
{
    // create the MPC controller
    const int dimx = 2;
    const int dimu = 2;
    const int num_segments = 6;
    const int poly_order = 3;
    double tf = 3.0;
    casadi::DM y_ref = casadi::DM(1.0);

    casadi::DMDict mpc_props;
    mpc_props["mpc.Q"] = casadi::DM(100);
    mpc_props["mpc.R"] = casadi::DM::diag(casadi::DM(std::vector<double>{0.5,0}));
    mpc_props["mpc.P"] = casadi::DM(100);

    polympc::nmpc<System, dimx, dimu, num_segments, poly_order> mpc(y_ref, tf, mpc_props);

    /** set state and control constraints */
    DM lbu = DM(std::vector<double>{-10, 0.5});
    DM ubu = DM(std::vector<double>{10, 2});
    mpc.setLBU(lbu);
    mpc.setUBU(ubu);

    DM lbx = DM::vertcat({-DM::inf(), -5});
    DM ubx = DM::vertcat({ DM::inf(),  5});
    mpc.setLBX(lbx);
    mpc.setUBX(ubx);

    DM state = DM::vertcat({0, 0});
    mpc.computeControl(state);
    DM opt_ctl  = mpc.getOptimalControl();
    DM opt_traj = mpc.getOptimalTrajetory();

    std::cout << opt_ctl << "\n";
    std::cout << opt_traj << "\n";

    std::cout << "Path Error: " << mpc.getPathError() << "\n";

    return 0;
}
