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
// eigen
#include "utils.h"

namespace vurtis {

class CostBase {

 public:

  CostBase(Vector& x_ref, Vector& u_ref) : x_ref_(x_ref), u_ref_(u_ref) {}

  Vector x_ref_; // state reference
  Vector u_ref_; // input reference

  VectorAD cost_eval_;

  virtual VectorAD LeastSquareCost(VectorAD &state, VectorAD &input, Vector &state_ref, Vector &input_ref) = 0;

  Matrix GradientCost(VectorAD &state, VectorAD &input, Vector &state_ref, Vector &input_ref) {
    return autodiff::jacobian([&](VectorAD &state,
                                           VectorAD &input,
                                           Vector &state_ref,
                                           Vector &input_ref) { return LeastSquareCost(state, input, state_ref, input_ref); },
                                       wrt(state, input),
                                       at(state, input, state_ref, input_ref),
                                       cost_eval_);
  }

 // void SetXref(const Vector &x_ref) { x_ref_ = x_ref; }
 // void SetUref(const Vector &u_ref) { u_ref_ = u_ref; }

};

struct ProblemInit {

  const size_t nx; // number of states of dynamic model
  const size_t nu; // number of inputs
  const size_t nz; // dimension of the least square cost function
  const size_t nh; // number of constraints (step=0,...,N-1)
  const size_t nh_e; // number of end-constraints (step=N)
  const size_t N; // number of steps
  const size_t num_parameters; // number of parameters

  Vector x0; // initial state


};

}