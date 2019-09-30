#ifndef DOUBLE_INTEGRATOR_MODEL_HPP
#define DOUBLE_INTEGRATOR_MODEL_HPP

#include "Eigen/Dense"

template <typename _Scalar = double>
struct System
{
    System(){A << 0,0,0, 0,1,0, 0,0,0; B << 1,0,0, 0,0,0, 0,0,-1; c << 0,0,0.25;}
    ~System(){}

    using Scalar     = _Scalar;
    using State      = Eigen::Matrix<Scalar, 3, 1>; // x1, x2, e
    using Control    = Eigen::Matrix<Scalar, 3, 1>; // u, D, v
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    Eigen::Matrix<Scalar, State::RowsAtCompileTime, State::RowsAtCompileTime> A;
    Eigen::Matrix<Scalar, State::RowsAtCompileTime, Control::RowsAtCompileTime> B;
    Eigen::Matrix<Scalar, State::RowsAtCompileTime, 1> c;

    /** the one for automatic differentiation */
    template<typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    void operator() (const Eigen::MatrixBase<DerivedA> &state, const Eigen::MatrixBase<DerivedB> &control,
                     const Eigen::MatrixBase<DerivedC> &param, Eigen::MatrixBase<DerivedD> &value) const
    {
        value = control[1] * (A * state + B * control + c);
    }
};

template<typename _Scalar = double>
struct Lagrange
{
    using Scalar = _Scalar;
    using State = Eigen::Matrix<Scalar, 3, 1>;
    using Control  = Eigen::Matrix<Scalar, 3, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    Eigen::Matrix<Scalar, 1, 1> Q;
    Eigen::Matrix<Scalar, Control::RowsAtCompileTime, Control::RowsAtCompileTime> R;
    Eigen::Matrix<Scalar, 1, 3> C;

    Lagrange(){
        Q << 100;
        R << 0.5, 0, 0,
              0,  0, 0,
              0,  0, 0;

        C << 0, 1, 0;
    }
    ~Lagrange(){}

    /** the one for automatic differentiation */
    template<typename DerivedA, typename DerivedB, typename DerivedC, typename CostT>
    void operator() (const Eigen::MatrixBase<DerivedA> &state, const Eigen::MatrixBase<DerivedB> &control,
                     const Eigen::MatrixBase<DerivedC> &param, CostT &value) const
    {
        // param = y_ss
        value = (C * state - param).dot(Q * (C * state - param)) + control.dot(R * control) - 1e-3 * state[2];
    }

};

template<typename _Scalar = double>
struct Mayer
{
    Mayer(){
        Q << 100;
        y_ss << 1.0;
        C << 0, 1, 0;
    }
    ~Mayer(){}

    using Scalar = _Scalar;
    using State = Eigen::Matrix<Scalar, 3, 1>;
    using Control  = Eigen::Matrix<Scalar, 3, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    Eigen::Matrix<Scalar, 1, 1> Q;
    Eigen::Matrix<Scalar, 1, 3> C;
    Eigen::Matrix<Scalar, 1, 1> y_ss;

    template<typename StateT, typename CostT>
    void operator() (const Eigen::MatrixBase<StateT> &state, CostT &value) const
    {
        using ScalarT = typename Eigen::MatrixBase<StateT>::Scalar;
        value = (C * state - y_ss).dot(Q.template cast<ScalarT>() * (C * state - y_ss));
    }
};



#endif // DOUBLE_INTEGRATOR_MODEL_HPP
