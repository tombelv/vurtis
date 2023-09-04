#pragma once

#include "utils.h"
#include "parameters_base.h"

namespace vurtis {

    enum CostType {LINEAR_LS, NONLINEAR_LS};

    template <typename Cost>
    class CostBase {

    public:

        CostBase(const Vector& x_ref, const Vector& u_ref) : x_ref_(x_ref), u_ref_(u_ref) {}

        Vector x_ref_;
        Vector u_ref_;

        VectorAD cost_eval_;
        VectorAD cost_eval_term_;

        const CostType cost_type = Cost::cost_type;
        const CostType term_cost_type = Cost::term_cost_type;

        VectorAD LeastSquareCostInterface(VectorAD &state, VectorAD &input, const Vector &state_ref, const Vector &input_ref, int t_idx) {
          return static_cast<Cost*>(this)->LeastSquareCost(state, input, state_ref, input_ref, t_idx);
        }
        VectorAD LeastSquareCostTerminalInterface(VectorAD &state, const Vector &state_ref, int t_idx) {
          return static_cast<Cost*>(this)->LeastSquareCostTerminal(state, state_ref, t_idx);
        }

        VectorAD EvalCost(VectorAD &state, VectorAD &input, const Vector &state_ref, const Vector &input_ref, int t_idx) {
          return static_cast<Cost*>(this)->LeastSquareCost(state, input, state_ref, input_ref, t_idx);
        }
        VectorAD EvalCostTerminal(VectorAD &state, const Vector &state_ref, int t_idx) {
          return static_cast<Cost*>(this)->LeastSquareCostTerminal(state, state_ref, t_idx);
        }


        Matrix GradientCost(VectorAD &state, VectorAD &input, Vector &state_ref, Vector &input_ref,const int t_idx) {
          return autodiff::jacobian([&](VectorAD &state,
                                        VectorAD &input,
                                        Vector &state_ref,
                                        Vector &input_ref,
                                        const int t_idx) {
                                        return LeastSquareCostInterface(state, input, state_ref, input_ref, t_idx); },
                                    wrt(state, input),
                                    at(state, input, state_ref, input_ref, t_idx),
                                    cost_eval_);
        }

        Matrix GradientCostTerminal(VectorAD &state, Vector &state_ref,const int t_idx) {
          return autodiff::jacobian([&](VectorAD &state,
                                        Vector &state_ref,
                                        const int t_idx) { return LeastSquareCostTerminalInterface(state, state_ref, t_idx); },
                                    wrt(state),
                                    at(state, state_ref, t_idx),
                                    cost_eval_term_);
        }


    };



}