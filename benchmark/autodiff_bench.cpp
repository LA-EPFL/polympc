#include <iostream>
#include <cmath>
#include "timer.hpp"

#define EIGEN_STACK_ALLOCATION_LIMIT 2097152

#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>


template <typename Scalar, int N=2>
Scalar rosenbrock(const Eigen::Matrix<Scalar, N, 1> x)
{
    const double a = 1;
    const double b = 100;

    Scalar z = 0;
    for (int i = 0; i < N-1; i++) {
        z += pow(a - x[i], 2) + b * pow(x[i+1] - pow(x[i], 2), 2);
    }
    return z;
}

template <typename ADVec>
void ADInit(ADVec &x)
{
    for (int i=0; i<x.rows(); i++) {
        x[i].derivatives().coeffRef(i) = 1;
    }
}

template <int N, typename Scalar=double>
void Eigen_benchmark()
{
    using grad_t = Eigen::Matrix<Scalar, N, 1>;
    using ADScalar = Eigen::AutoDiffScalar<grad_t>;
    using ADVec = Eigen::Matrix<ADScalar, N, 1>;

    const int NAVG = 100;

    ADScalar res;
    ADVec x;
    double m1, m2, s1, s2;
    Timer t;

    // eval + diff
    x.fill(0.0);
    for (int i = 0; i < NAVG; i++) {
        t.tic();
        ADInit(x);
        res = rosenbrock<ADScalar, N>(x);
        t.toc();
    }
    // t.print();
    std::tie(m1, s1) = t.mean_std();
    t.clear();


    // eval
    Eigen::Matrix<Scalar, N, 1> x2;
    x2.fill(0.0);
    for (int i = 0; i < NAVG; i++) {
        t.tic();
        rosenbrock<Scalar, N>(x2);
        t.toc();
    }
    // t.print();
    std::tie(m2, s2) = t.mean_std();
    t.clear();

    // std::cout << "res.derivatives() " << res.derivatives().transpose() << std::endl;
    printf("[%5d, %8.3f, %8.3f, %8.3f, %8.3f, %9lu],\n", N, m1, s1, m2, s2, sizeof(x));
}

#include "casadi/casadi.hpp"
using namespace casadi;

SX rosenbrock(SX &x)
{

    const double a = 1;
    const double b = 100;

    SX z = 0;
    for (int i = 0; i < x.rows()-1; i++) {
        z += pow(a - x(i), 2) + b * pow(x(i+1) - pow(x(i), 2), 2);
    }
    return z;
}


void casadi_benchmark(int N)
{
    SX x = SX::sym("x", N);

    SX rb = rosenbrock(x);
    SX rb_grad = SX::gradient(rb,x);

    Function rbf = Function("rb", {x}, {rb});
    Function rb_gradf = Function("rb_grad", {x}, {rb_grad});

    const int NAVG = 100;
    DMVector grad;
    DM val;
    DM x0 = DM::zeros(N,1);

    double m1, m2, s1, s2;
    Timer t;

    for (int i = 0; i < NAVG; i++) {
        t.tic();
        grad = rb_gradf({x0});
        t.toc();
    }
    // t.print();
    std::tie(m1, s1) = t.mean_std();
    t.clear();

    for (int i = 0; i < NAVG; i++) {
        t.tic();
        val = rbf({x0});
        t.toc();
    }
    // t.print();
    std::tie(m2, s2) = t.mean_std();
    t.clear();

    printf("[%5d, %8.3f, %8.3f, %8.3f, %8.3f],\n", N, m1, s1, m2, s2);
}

void casadi_benchmark_codegen(int N)
{
    SX x = SX::sym("x", N);

    SX rb = rosenbrock(x);
    SX rb_grad = SX::gradient(rb,x);

    Function rbf = Function("rb", {x}, {rb});
    Function rb_gradf = Function("rb_grad", {x}, {rb_grad});

    CodeGenerator codegen = CodeGenerator("gen.c");
    codegen.add(rbf);
    codegen.add(rb_gradf);
    codegen.generate();

    const int NAVG = 100;
    DMVector grad;
    DM val;
    DM x0 = DM::zeros(N,1);

    Importer C = Importer("gen.c","shell");
    Function gen_rb = external("rb", C);
    Function gen_rb_grad = external("rb_grad", C);

    double m1, m2, s1, s2;
    Timer t;

    for (int i = 0; i < NAVG; i++) {
        t.tic();
        grad = gen_rb_grad({x0});
        t.toc();
    }
    // t.print();
    std::tie(m1, s1) = t.mean_std();
    t.clear();

    for (int i = 0; i < NAVG; i++) {
        t.tic();
        val = gen_rb({x0});
        t.toc();
    }
    // t.print();
    std::tie(m2, s2) = t.mean_std();
    t.clear();

    printf("[%5d, %8.3f, %8.3f, %8.3f, %8.3f],\n", N, m1, s1, m2, s2);
}

int main(int argc, char *argv[])
{
    printf("Eigen AutoDiff\n");
    printf("N       grad[us]   std[us]  eval[us]    std[us]  sizeof(x)\n");
    Eigen_benchmark<2>();
    Eigen_benchmark<5>();
    Eigen_benchmark<10>();
    Eigen_benchmark<20>();
    Eigen_benchmark<50>();
    Eigen_benchmark<100>();
    Eigen_benchmark<200>();
    Eigen_benchmark<500>();

    printf("Eigen AutoDiff float\n");
    printf("N       grad[us]   std[us]  eval[us]    std[us]  sizeof(x)\n");
    Eigen_benchmark<2,float>();
    Eigen_benchmark<5,float>();
    Eigen_benchmark<10,float>();
    Eigen_benchmark<20,float>();
    Eigen_benchmark<50,float>();
    Eigen_benchmark<100,float>();
    Eigen_benchmark<200,float>();
    Eigen_benchmark<500,float>();

    printf("CasADi\n");
    printf("N       grad[us]   std[us]  eval[us]   std[us]\n");
    casadi_benchmark(2);
    casadi_benchmark(5);
    casadi_benchmark(10);
    casadi_benchmark(20);
    casadi_benchmark(50);
    casadi_benchmark(100);
    casadi_benchmark(200);
    casadi_benchmark(500);
    casadi_benchmark(1000);
    casadi_benchmark(2000);
    casadi_benchmark(5000);
    casadi_benchmark(10000);

    // printf("CasADi codegen\n");
    // printf("N      grad[us]  eval[us]\n");
    // casadi_benchmark_codegen(2);
    // casadi_benchmark_codegen(5);
    // casadi_benchmark_codegen(10);
    // casadi_benchmark_codegen(20);
    // casadi_benchmark_codegen(50);
    // casadi_benchmark_codegen(100);
    // casadi_benchmark_codegen(200);
    // casadi_benchmark_codegen(500);
    // casadi_benchmark_codegen(1000);
    // casadi_benchmark_codegen(2000);
    // casadi_benchmark_codegen(5000);
    // casadi_benchmark_codegen(10000);
}
