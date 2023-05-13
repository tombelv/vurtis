#pragma once
// eigen
#include "utils.h"

namespace vurtis {

class CostBase {

 public:

  CostBase(const Vector& x_ref, const Vector& u_ref) : x_ref_(x_ref), u_ref_(u_ref) {}

  Vector x_ref_; // state reference
  Vector u_ref_; // input reference

  VectorAD cost_eval_;
  VectorAD cost_eval_term_;

  virtual VectorAD LeastSquareCost(VectorAD &state, VectorAD &input, const Vector &state_ref, const Vector &input_ref, const Vector & nlp_params) = 0;
  virtual VectorAD LeastSquareCostTerminal(VectorAD &state, const Vector &state_ref, const Vector & nlp_params) = 0;


    Matrix GradientCost(VectorAD &state, VectorAD &input, Vector &state_ref, Vector &input_ref, const Vector & nlp_params) {
    return autodiff::jacobian([&](VectorAD &state,
                                           VectorAD &input,
                                           Vector &state_ref,
                                           Vector &input_ref,
                                           const Vector & nlp_params) {
                                        return LeastSquareCost(state, input, state_ref, input_ref, nlp_params); },
                                       wrt(state, input),
                                       at(state, input, state_ref, input_ref, nlp_params),
                                       cost_eval_);
  }

    Matrix GradientCostTerminal(VectorAD &state, Vector &state_ref, const Vector & nlp_params) {
      return autodiff::jacobian([&](VectorAD &state,
                                    Vector &state_ref,
                                    const Vector & nlp_params) { return LeastSquareCostTerminal(state, state_ref, nlp_params); },
                                wrt(state),
                                at(state, state_ref, nlp_params),
                                cost_eval_term_);
    }

 // void SetXref(const Vector &x_ref) { x_ref_ = x_ref; }
 // void SetUref(const Vector &u_ref) { u_ref_ = u_ref; }

};

struct ProblemInit {

  const int nx; // number of states of dynamic model
  const int nu; // number of inputs
  const int nz; // dimension of the least square cost function
  const int nh; // number of constraints (step=0,...,N-1)
  const int nh_e; // number of end-constraints (step=N)
  const int N; // number of steps

  Matrix parameters; // initialization of parameters
  Vector x0; // initial state


};

}