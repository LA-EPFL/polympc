#ifndef DOUBLE_INTEGRATOR_MODEL_HPP
#define DOUBLE_INTEGRATOR_MODEL_HPP

#include "Eigen/Dense"

template <typename _Scalar = double>
struct System
{
    System(){A << 0, 0, 1, 0; B << 1, 0;}
    ~System(){}

    using Scalar     = _Scalar;
    using State      = Eigen::Matrix<Scalar, 2, 1>;
    using Control    = Eigen::Matrix<Scalar, 2, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    Eigen::Matrix<Scalar, State::RowsAtCompileTime, State::RowsAtCompileTime> A;
    Eigen::Matrix<Scalar, State::RowsAtCompileTime, 1> B;

    /** the one for automatic differentiation */
    template<typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    void operator() (const Eigen::MatrixBase<DerivedA> &state, const Eigen::MatrixBase<DerivedB> &control,
                     const Eigen::MatrixBase<DerivedC> &param, Eigen::MatrixBase<DerivedD> &value) const
    {
        value = control[1] * (A * state + B * control[0]);
    }
};

template<typename _Scalar = double>
struct Lagrange
{
    using Scalar = _Scalar;
    using State = Eigen::Matrix<Scalar, 2, 1>;
    using Control  = Eigen::Matrix<Scalar, 2, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    Eigen::Matrix<Scalar, 1, 1> Q;
    Eigen::Matrix<Scalar, Control::RowsAtCompileTime, Control::RowsAtCompileTime> R;
    Eigen::Matrix<Scalar, 1, 2> C;
    Eigen::Matrix<Scalar, 1, 1> y_ss;

    Lagrange(){
        Q << 100;
        R << 0.5, 0,
             0, 0;
        y_ss << 1.0;
        C << 0, 1;
    }
    ~Lagrange(){}

    /** the one for automatic differentiation */
    template<typename DerivedA, typename DerivedB, typename DerivedC, typename CostT>
    void operator() (const Eigen::MatrixBase<DerivedA> &state, const Eigen::MatrixBase<DerivedB> &control,
                     const Eigen::MatrixBase<DerivedC> &param, CostT &value) const
    {
        value = (C * state - y_ss).dot(Q * (C * state - y_ss)) + control.dot(R * control);
    }

};

template<typename _Scalar = double>
struct Mayer
{
    Mayer(){
        Q << 100;
        y_ss << 1.0;
        C << 0, 1;
    }
    ~Mayer(){}

    using Scalar = _Scalar;
    using State = Eigen::Matrix<Scalar, 2, 1>;
    using Control  = Eigen::Matrix<Scalar, 2, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    Eigen::Matrix<Scalar, 1, 1> Q;
    Eigen::Matrix<Scalar, 1, 2> C;
    Eigen::Matrix<Scalar, 1, 1> y_ss;

    template<typename StateT, typename CostT>
    void operator() (const Eigen::MatrixBase<StateT> &state, CostT &value) const
    {
        using ScalarT = typename Eigen::MatrixBase<StateT>::Scalar;
        value = (C * state - y_ss).dot(Q.template cast<ScalarT> * (C * state - y_ss));
    }
};



#endif // DOUBLE_INTEGRATOR_MODEL_HPP
