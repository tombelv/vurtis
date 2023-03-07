# Very Uncomplicated Real Time Iteration Scheme (vurtis)
This repository contains a simple header-only implementation of the RTI scheme
in a multiple-shooting fashion using the sparse [OSQP](https://osqp.org/) solver and [autodiff](https://github.com/autodiff/autodiff/) for algorithmic differentiation.

The goal is to provide a fast - yet simple - C++ NMPC library. While care has been taken to optimize performance, surely
much more could be done.

Everything is still experimental and breaking changes might occur.

It can be used by including it as an INTERFACE library, see examples.

## Requirements
* Eigen3
* [OsqpEigen](https://github.com/robotology/osqp-eigen)

and their dependencies.

The autodiff header-only library is bundled in the `include/` directory.


## Usage
The user is required to define the implementation of the system dynamics, constraints and cost function
via the two classes `vurtis::Model` and `vurtis::Cost` which inherit the additional necessary functionality
from the abstract classes `vurtis::ModelBase` and `vurtis::CostBase` respectively.

The numerical integration is performed using 4-th order Runge-Kutta method
and all the sensitivities are automatically computed
using AD.

It is possible to define step-wise parameters for the constraints.

### Example
The provided `examples/simple_example.cpp` performs trajectory tracking with a kinematic unicycle vehicle 
while satisfying input constraints.

## TODO
- [ ] implement terminal cost
- [ ] include autodiff as a submodule or system dependency
- [ ] some interface to set OSQP settings during setup
- [ ] maybe rewrite the OSQP wrapper to avoid unnecessary checks


