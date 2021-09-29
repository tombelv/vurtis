// vurtis
//
// MIT License
//
// Copyright (c) 2021 Tommaso Belvedere
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#pragma once

#include "utils.h"

namespace vurtis {

class ModelBase {

 private:
  const double dt_;

 public:

  ModelBase(double dt) : dt_{dt} { }


  // storing evaluation from sensitivity computed with forward AD
  VectorAD F_eval_;
  VectorAD h_eval_;
  VectorAD he_eval_;

  //------------------------------------------------------------------------------------------------------------------

  // continuous-time Dynamics to be implemented
  virtual VectorAD Dynamics(VectorAD &state, VectorAD &input) = 0;

  // discretization with Runge-Kutta 4th order
  // ns steps over the sampling time interval dt_

  VectorAD step(VectorAD &state, VectorAD &input) {

    VectorAD state_next = state;

    size_t ns = 1;
    double h = dt_ / ns;

    for (size_t i = 0; i < ns; ++i) {
      VectorAD k1 = Dynamics(state, input);

      VectorAD temp1 = state + 0.5 * k1 * h;
      VectorAD k2 = Dynamics(temp1, input);

      VectorAD temp2 = state + 0.5 * k2 * h;
      VectorAD k3 = Dynamics(temp2, input);

      VectorAD temp3 = state + k3 * h;
      VectorAD k4 = Dynamics(temp3, input);

      state_next += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
    }

    return state_next;
  }

  // Integrator interface for plain Eigen vectors
  Vector integrator(const Vector &state_,const Vector &input_) {
    VectorAD state = state_;
    VectorAD input = input_;

    VectorAD state_next = step(state, input);

    return state_next.cast<double>();

  }

  // Dynamics sensitivities
  // lambda functions are needed to pass member functions to autodiff
  Matrix Ad(VectorAD &state, VectorAD &input) {
    return autodiff::jacobian([&](VectorAD &state, VectorAD &input) {
      return step(state,
                  input);
    }, wrt(state), at(state, input), F_eval_);
  }

  Matrix Bd(VectorAD &state, VectorAD &input) {
    return autodiff::jacobian([&](VectorAD &state, VectorAD &input) {
      return step(state,
                  input);
    }, wrt(input), at(state, input), F_eval_);
  }
  //------------------------------------------------------------------------------------------------------------------

  // constraints (to be implemented) and their sensitivities

  virtual VectorAD Constraint(VectorAD &state, VectorAD &input, Eigen::VectorXd &params) = 0;
  virtual VectorAD EndConstraint(VectorAD &state, Eigen::VectorXd &params) = 0;

  Matrix Cd(VectorAD &state, VectorAD &input, Eigen::VectorXd &params) {
    return autodiff::jacobian([&](VectorAD &state,
                                           VectorAD &input,
                                           Eigen::VectorXd &params) { return Constraint(state, input, params); },
                                       wrt(state),
                                       at(state, input, params),
                                       h_eval_);
  }

  Matrix Dd(VectorAD &state, VectorAD &input, Eigen::VectorXd &params) {
    return autodiff::jacobian([&](VectorAD &state,
                                           VectorAD &input,
                                           Eigen::VectorXd &params) { return Constraint(state, input, params); },
                                       wrt(input),
                                       at(state, input, params),
                                       h_eval_);
  }



  Matrix Cd_e(VectorAD &state, Eigen::VectorXd &params) {
    return autodiff::jacobian([&](VectorAD &state,
                                           Eigen::VectorXd &params) { return EndConstraint(state, params); },
                                       wrt(state),
                                       at(state, params),
                                       he_eval_);
  }
  //------------------------------------------------------------------------------------------------------------------

};

}