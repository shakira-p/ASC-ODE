#include <iostream>
#include <autodiff.hpp>
#include <nonlinfunc.hpp>
#include <vector>
#include <iomanip>

using namespace ASC_ode;

template <typename T>
void LegendrePolynomials(int n, T x, std::vector<T>& P) {
  if (n < 0) {
    P.clear();
    return;
  }
  P.resize(n + 1);
  P[0] = T(1);
  if (n == 0) return;
  P[1] = x;
  for (int k = 2; k <= n; ++k) {
    P[k] = ((T(2 * k - 1) * x * P[k - 1]) - T(k - 1) * P[k - 2]) / T(k);
  }
}

// ============================================================================
// PendulumAD class: Models a simple pendulum using automatic differentiation
// Second-order ODE: alpha'' = -g/l * sin(alpha)
// Reformulated as first-order system in R^2:
//   y0 = alpha (angular displacement)  // =alpha(t_0)
//   y1 = alpha' (angular velocity)  // =alpha_0_p
//   y0' = y1
//   y1' = -g/l * sin(y0)
// ============================================================================

class PendulumAD : public NonlinearFunction
{
private:
  double m_length;
  double m_gravity;

public:
  PendulumAD(double length, double gravity=9.81) : m_length(length), m_gravity(gravity) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }

  void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    T_evaluate<double>(x, f);
  }

  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    Vector<AutoDiff<2>> x_ad(2);
    Vector<AutoDiff<2>> f_ad(2);

    x_ad(0) = Variable<0>(x(0));
    x_ad(1) = Variable<1>(x(1));
    T_evaluate<AutoDiff<2>>(x_ad, f_ad);

    for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 2; j++)
        df(i,j) = f_ad(i).deriv()[j];
  }

  template <typename T>
  void T_evaluate (VectorView<T> x, VectorView<T> f) const

  {
    f(0) = x(1);
    f(1) = -m_gravity/m_length*sin(x(0));
  }
};

int main(int argc, char* argv[])
{
  // Evaluate and plot Legendre-polynomials up to order 5, in the interval -1<=x<=1
  // Evaluate and plot also their derivatives (using AutoDiff)

  std::cout << "x,P0,P0_prime,P1,P1_prime,P2,P2_prime,P3,P3_prime,P4,P4_prime,P5,P5_prime" << std::endl;

  for (double x = -1.0; x <= 1.0; x += 0.02) {
    AutoDiff<1> adx = Variable<0>(x);
    std::vector<AutoDiff<1>> P;
    LegendrePolynomials(5, adx, P);

    // Output: x, P_0(x), P_0'(x), P_1(x), P_1'(x), ..., P_5(x), P_5'(x)
    std::cout << x;
    for (int n = 0; n <= 5; ++n) {
      std::cout << "," << P[n].value() << "," << P[n].deriv()[0];
    }
    std::cout << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  return 0;
}

