#include <iostream>
#include <cmath>
#include <autodiff.hpp>
#include <nonlinfunc.hpp>
#include <timestepper.hpp>
#include <vector.hpp>

using namespace ASC_ode;
using namespace nanoblas;

constexpr double g = 9.81;  // gravitational acceleration
constexpr double l = 1.0;   // pendulum length

// Template function for right-hand side evaluation
// State: [alpha, alpha_dot]
// ODE: alpha' = alpha_dot
//      alpha_dot' = -(g/l) * sin(alpha)
template <typename T>
void T_evaluate(const T& alpha, const T& alpha_dot, T& f0, T& f1)
{
    f0 = alpha_dot;
    f1 = -(g / l) * sin(alpha);
}

class PendulumFunction : public NonlinearFunction
{
public:
    virtual size_t dimX() const override { return 2; }
    virtual size_t dimF() const override { return 2; }

    virtual void evaluate(VectorView<double> x, VectorView<double> f) const override
    {
        double alpha = x(0);
        double alpha_dot = x(1);
        
        T_evaluate(alpha, alpha_dot, f(0), f(1));
    }

    virtual void evaluateDeriv(VectorView<double> x, MatrixView<double> df) const override
    {
        AutoDiff<2> alpha(Variable<0>(x(0)));
        AutoDiff<2> alpha_dot(Variable<1>(x(1)));
        
        AutoDiff<2> f0, f1;
        T_evaluate(alpha, alpha_dot, f0, f1);
        
        // Fill Jacobian matrix: df[i][j] = df_i/dx_j
        df(0, 0) = f0.deriv()[0];  // df0/dalpha
        df(0, 1) = f0.deriv()[1];  // df0/dalpha_dot
        df(1, 0) = f1.deriv()[0];  // df1/dalpha
        df(1, 1) = f1.deriv()[1];  // df1/dalpha_dot
    }
};

int main()
{
    std::cout << "=== Pendulum AutoDiff Test ===" << std::endl;
    
    // Initial conditions
    double alpha_0 = M_PI / 4;      // 45 degrees
    double alpha_dot_0 = 0.0;       // released from rest
    
    Vector<> state(2);
    state(0) = alpha_0;
    state(1) = alpha_dot_0;
    
    std::cout << "\nInitial state: alpha = " << alpha_0 
              << ", alpha_dot = " << alpha_dot_0 << std::endl;

    PendulumFunction pendulum;
    Vector<> f(2);
    pendulum.evaluate(state, f);
    
    std::cout << "\nRight-hand side evaluation:" << std::endl;
    std::cout << "f = [" << f(0) << ", " << f(1) << "]" << std::endl;
    std::cout << "Expected: [" << alpha_dot_0 << ", " 
              << -(g/l) * std::sin(alpha_0) << "]" << std::endl;

    Matrix<> jacobian(2, 2);
    pendulum.evaluateDeriv(state, jacobian);
    
    std::cout << "\nJacobian matrix (using AutoDiff):" << std::endl;
    std::cout << "[ " << jacobian(0,0) << ", " << jacobian(0,1) << " ]" << std::endl;
    std::cout << "[ " << jacobian(1,0) << ", " << jacobian(1,1) << " ]" << std::endl;

    // df0/dalpha = 0, df0/dalpha_dot = 1
    // df1/dalpha = -(g/l)*cos(alpha), df1/dalpha_dot = 0
    std::cout << "\nExpected Jacobian (analytical):" << std::endl;
    std::cout << "[ " << 0.0 << ", " << 1.0 << " ]" << std::endl;
    std::cout << "[ " << -(g/l)*std::cos(alpha_0) << ", " << 0.0 << " ]" << std::endl;

    double tol = 1e-10;
    bool correct = true;
    
    if (std::abs(jacobian(0,0) - 0.0) > tol) correct = false;
    if (std::abs(jacobian(0,1) - 1.0) > tol) correct = false;
    if (std::abs(jacobian(1,0) + (g/l)*std::cos(alpha_0)) > tol) correct = false;
    if (std::abs(jacobian(1,1) - 0.0) > tol) correct = false;
    
    std::cout << "\n" << (correct ? "✓ Test PASSED" : "✗ Test FAILED") << std::endl;
    
    return correct ? 0 : 1;
}
