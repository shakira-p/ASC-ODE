#include <iostream>
#include <fstream> 

#include <nonlinfunc.hpp>
#include <timestepper.hpp>
#include <implicitRK.hpp>

using namespace ASC_ode;


class MassSpring : public NonlinearFunction
{
private:
  double mass;
  double stiffness;

public:
  MassSpring(double m, double k) : mass(m), stiffness(k) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }
  
  void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    f(0) = x(1);
    f(1) = -stiffness/mass*x(0);
  }
  
  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    df = 0.0;
    df(0,1) = 1;
    df(1,0) = -stiffness/mass;
  }
};


int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cout << "Usage: " << argv[0] << " <steps>" << std::endl;
    return 1;
  }

  double tend = 4*M_PI;

  // in oder to not build it each time when changing stepsize, we make it an argument
  const int steps = std::stoi(argv[1]);  // get nr steps as argument

  double tau = tend/steps;

  Vector<> y = { 1, 0 };  // initializer list
  auto rhs = std::make_shared<MassSpring>(1.0, 1.0);



/*
  Vector<> Radau(3), RadauWeight(3);
  GaussRadau (Radau, RadauWeight);
  // not sure about weights, comput them via ComputeABfromC
  cout << "Radau = " << Radau << ", weight = " << RadauWeight <<  endl;
        Vector<> Gauss2c(2), Gauss3c(3);
*/


  // ExplicitEuler stepper(rhs);
  // ImplicitEuler stepper(rhs);
  // ImprovedEuler stepper(rhs);
  CrankNicolson stepper(rhs);

  // RungeKutta stepper(rhs, Gauss2a, Gauss2b, Gauss2c);

  // Gauss3c .. points tabulated, compute a,b:
  auto [Gauss3a,Gauss3b] = ComputeABfromC (Gauss3c);
  //ImplicitRungeKutta stepper(rhs, Gauss3a, Gauss3b, Gauss3c);


  /*
  // arbitrary order Gauss-Legendre
  int stages = 5;
  Vector<> c(stages), b1(stages);
  GaussLegendre(c, b1);

  auto [a, b] = ComputeABfromC(c);
  ImplicitRungeKutta stepper(rhs, a, b, c);
  */

  /*
  // arbitrary order Radau
  int stages = 5;
  Vector<> c(stages), b1(stages);
  GaussRadau(c, b1);

  auto [a, b] = ComputeABfromC(c);
  ImplicitRungeKutta stepper(rhs, a, b, c);
  */


  std::ofstream outfile ("../ASC-ODE/demos/data/crank.txt");
  std::cout << 0.0 << "  " << y(0) << " " << y(1) << " " << steps << std::endl;
  outfile << 0.0 << "  " << y(0) << " " << y(1) << " " << steps << std::endl;

  for (int i = 0; i < steps; i++)
  {
     stepper.DoStep(tau, y);

     std::cout << (i+1) * tau << "  " << y(0) << " " << y(1) << " " << steps << std::endl;
     outfile << (i+1) * tau << "  " << y(0) << " " << y(1) << " " <<  steps << std::endl;
  }
}
