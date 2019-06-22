#ifndef SIMPLE_ROBOT_MODEL_HPP
#define SIMPLE_ROBOT_MODEL_HPP

#include "Eigen/Dense"
#include "Eigen/Sparse"

#include <iostream>
#include <fstream>
#include <Eigen/Dense>

Eigen::MatrixXd readCSV(std::string file, int rows, int cols)
{
    std::ifstream in(file);
    std::string line;

    int row = 0;
    int col = 0;

    Eigen::MatrixXd res = Eigen::MatrixXd(rows, cols);

    if (in.is_open()) {
        while (std::getline(in, line)) {

            char *ptr = (char *) line.c_str();
            int len = line.length();
            col = 0;

            char *start = ptr;
            for (int i = 0; i < len; i++) {
                if (ptr[i] == ',') {
                    res(row, col++) = atof(start);
                    start = ptr + i + 1;
                }
            }
            res(row, col) = atof(start);
            row++;
        }
        in.close();
    }
    return res;
}

template <typename _Scalar = double>
struct MobileRobot
{
    MobileRobot(){}
    ~MobileRobot(){}

    using Scalar     = _Scalar;
    using State      = Eigen::Matrix<Scalar, 3, 1>;
    using Control    = Eigen::Matrix<Scalar, 2, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    /** the one for automatic differentiation */
    template<typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    void operator() (const Eigen::MatrixBase<DerivedA> &state, const Eigen::MatrixBase<DerivedB> &control,
                     const Eigen::MatrixBase<DerivedC> &param, Eigen::MatrixBase<DerivedD> &value) const
    {
        value[0] = control[0] * cos(state[2]) * cos(control[1]);
        value[1] = control[0] * sin(state[2]) * cos(control[1]);
        value[2] = control[0] * sin(control[1]) / param[0];
    }
};

template<typename _Scalar = double>
struct Lagrange
{
    using Scalar = _Scalar;
    using State = Eigen::Matrix<Scalar, 3, 1>;
    using Control  = Eigen::Matrix<Scalar, 2, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    Eigen::Matrix<Scalar, State::RowsAtCompileTime, State::RowsAtCompileTime> Q;
    Eigen::Matrix<Scalar, Control::RowsAtCompileTime, Control::RowsAtCompileTime> R;

    Lagrange(){
        Q << 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.01;
        R << 1, 0, 0, 0.001;
        Q = readCSV("../LQ.csv",3,3);
        R = readCSV("../LR.csv",2,2);
    }
    ~Lagrange(){}


    /** the one for automatic differentiation */
    template<typename DerivedA, typename DerivedB, typename DerivedC, typename CostT>
    void operator() (const Eigen::MatrixBase<DerivedA> &state, const Eigen::MatrixBase<DerivedB> &control,
                     const Eigen::MatrixBase<DerivedC> &param, CostT &value) const
    {
        //value = state.dot(Q * state) + control.dot(R * control);
        /** @note: does not work with second derivatives without explicit cast ???*/
        using ScalarT = typename Eigen::MatrixBase<DerivedA>::Scalar;
        value = state.dot(Q.template cast<ScalarT>() * state) + control.dot(R. template cast<ScalarT>() * control);
    }

};

template<typename _Scalar = double>
struct Mayer
{
    Mayer(){
        Q << 20, 0, 0, 0, 20, 0, 0, 0, 10;
        Q = readCSV("../MQ.csv",3,3);
    }
    ~Mayer(){}

    using Scalar = _Scalar;
    using State = Eigen::Matrix<Scalar, 3, 1>;
    using Control  = Eigen::Matrix<Scalar, 2, 1>;
    using Parameters = Eigen::Matrix<Scalar, 1, 1>;

    Eigen::Matrix<Scalar, State::RowsAtCompileTime, State::RowsAtCompileTime> Q;

    template<typename StateT, typename CostT>
    void operator() (const Eigen::MatrixBase<StateT> &state, CostT &value) const
    {
        using ScalarT = typename Eigen::MatrixBase<StateT>::Scalar;
        value = state.dot(Q.template cast<ScalarT>() * state);
    }
};

#endif // SIMPLE_ROBOT_MODEL_HPP
