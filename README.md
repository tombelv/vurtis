# Very Uncomplicated Real Time Iteration Scheme (vurtis)
This repository contains a simple header-only implementation of the RTI scheme
in a multiple-shooting fashion using the sparse [OSQP](https://osqp.org/) solver and [autodiff](https://github.com/autodiff/autodiff/) for algorithmic differentiation.

The goal is to provide a fast - yet simple - C++ NMPC library. While care has been taken to optimize performance, surely
much more could be done.

For now this works as a template. Everything is still experimental and breaking changes might occur.


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
The provided `src/simple_example.cpp` performs trajectory tracking with a kinematic unicycle vehicle 
while satisfying input constraints.

## TODO
- [ ] extend the cost function hessian to the full nonlinear case
- [ ] implement terminal cost
- [ ] include autodiff as a submodule or system dependency
- [ ] some interface to set OSQP settings during setup
- [ ] rewrite OSQP interface for more performance

# License

MIT License

Copyright (c) 2022 Tommaso Belvedere

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.