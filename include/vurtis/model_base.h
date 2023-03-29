#pragma once

#include "utils.h"

namespace vurtis {

    class ModelBase {


    public:

        ModelBase(double dt, int nx, int nu, int nh, int nh_e) : dt_(dt), nx_(nx), nu_(nu), nh_(nh), nh_e_(nh_e) {}


        // storing evaluation from sensitivity computed with forward AD
        VectorAD F_eval_;
        VectorAD h_eval_;
        VectorAD he_eval_;

        const double dt_;
        const int nx_;
        const int nu_;
        const int nh_;
        const int nh_e_;

        //------------------------------------------------------------------------------------------------------------------

        // continuous-time Dynamics to be implemented
        virtual VectorAD Dynamics(VectorAD &state, VectorAD &input) = 0;

        virtual VectorAD Dynamics(VectorAD &state, VectorAD &input, VectorAD &params) {
          return Dynamics(state, input);
        }

        // discretization with Runge-Kutta 4th order
        // ns steps over the sampling time interval dt_

        VectorAD step(VectorAD &state, VectorAD &input) {

          VectorAD state_next = state;

          int ns = 1;
          double h = dt_ / ns;

          for (int i = 0; i < ns; ++i) {
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

        VectorAD step(VectorAD &state, VectorAD &input, VectorAD &params) {

          VectorAD state_next = state;

          int ns = 1;
          double h = dt_ / ns;

          for (int i = 0; i < ns; ++i) {
            VectorAD k1 = Dynamics(state, input, params);

            VectorAD temp1 = state + 0.5 * k1 * h;
            VectorAD k2 = Dynamics(temp1, input, params);

            VectorAD temp2 = state + 0.5 * k2 * h;
            VectorAD k3 = Dynamics(temp2, input, params);

            VectorAD temp3 = state + k3 * h;
            VectorAD k4 = Dynamics(temp3, input, params);

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
        // lambda functions are needed to pass member functions to autodiff as
        // suggested in https://github.com/autodiff/autodiff/issues/51
        Matrix Ad(VectorAD &state, VectorAD &input) {
          return autodiff::jacobian([&](VectorAD &state, VectorAD &input) {return step(state,input); },
                                    wrt(state), at(state, input), F_eval_);
        }

        Matrix Bd(VectorAD &state, VectorAD &input) {
          return autodiff::jacobian([&](VectorAD &state, VectorAD &input) {return step(state,input);},
                                    wrt(input), at(state, input), F_eval_);
        }

        virtual MatrixAD dFdxu(VectorAD &state, VectorAD &input, VectorAD &params) {
          MatrixAD res(nx_, nx_+nu_);
          res.leftCols(nx_) = Ad(state, input);
          res.rightCols(nu_) = Bd(state, input);

          return res;
        }

        //------------------------------------------------------------------------------------------------------------------

        // constraints (to be implemented) and their sensitivities

        virtual VectorAD Constraint(VectorAD &state, VectorAD &input, const Vector &params) = 0;
        virtual VectorAD EndConstraint(VectorAD &state, const Vector &params) = 0;

        Matrix Cd(VectorAD &state, VectorAD &input, const Vector &params) {
          return autodiff::jacobian([&](VectorAD &state, VectorAD &input, const Vector &params){return Constraint(state, input, params); },
                                    wrt(state),
                                    at(state, input, params),
                                    h_eval_);
        }

        Matrix Dd(VectorAD &state, VectorAD &input, const Vector &params) {
          return autodiff::jacobian([&](VectorAD &state, VectorAD &input, const Vector &params) { return Constraint(state, input, params); },
                                    wrt(input),
                                    at(state, input, params),
                                    h_eval_);
        }

        virtual MatrixAD dConstrdxu(VectorAD &state, VectorAD &input, VectorAD &params, const Vector &nlp_params) {
          MatrixAD res(nh_, nx_+nu_);
          res.leftCols(nx_) = Cd(state, input, nlp_params);
          res.rightCols(nu_) = Dd(state, input, nlp_params);

          return res;
        }


        Matrix Cd_e(VectorAD &state, Vector &params) {
          return autodiff::jacobian([&](VectorAD &state, Vector &params) { return EndConstraint(state, params); },
                                    wrt(state),
                                    at(state, params),
                                    he_eval_);
        }
        //------------------------------------------------------------------------------------------------------------------

        virtual Vector GetModelParams() const {};


    };

}