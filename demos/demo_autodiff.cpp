#include <iostream>
#include <autodiff.hpp>
#include <vector>


using namespace ASC_ode;


template <typename T>
T func1 (T x, T y)
{
  return x * sin(y);
  // return 1e6 + y;
}

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


int main()  // int argc, char* argv[]
{
  // if (argc != 2) {
  //   std::cout << "Usage: " << argv[0] << " <x>" << std::endl;
  //   return 1;
  // }
  // double x = std::stod(argv[1]);
  //
  // double y =

  double x = 1, y = 2;
  AutoDiff<2> adx = Variable<0>(x);
  AutoDiff<2> ady = Variable<1>(y);

  std::cout << "adx = " << adx << std::endl;
  std::cout << "ady = " << ady << std::endl;

  AutoDiff<2> prod = adx * ady;
  std::cout << "prod = " << prod << std::endl;

  std::cout << "func1(adx, ady) = " << func1(adx, ady) << std::endl;

  double eps = 1e-8;
  std::cout << "numdiff df/dx = " << (func1(x + eps, y) - func1(x-eps, y)) / (2*eps) << std::endl;
  std::cout << "numdiff df/dy = " << (func1(x, y + eps) - func1(x, y-eps)) / (2*eps) << std::endl;


  {
    // we can do second derivatives:
    AutoDiff<1, AutoDiff<1>> addx{Variable<0>(2)};
    std::cout << "addx = " << addx << std::endl;
    // func = x*x
    // func' = 2*x
    // func'' = 2
    std::cout << "addx*addx = " << addx * addx << std::endl;

    // std::cout << "sin(addx) = " << sin(addx) << std::endl;
  }
  return 0;

  // Evaluate and plot Legendre-polynomials up to order 5, in the interval -1<=x<=1
  // Evaluate and plot also their derivatives (using AutoDiff)

}