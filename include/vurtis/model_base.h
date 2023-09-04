#pragma once

#include "utils.h"

namespace vurtis {

    template<class Model>
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
        VectorAD DynamicsInterface(VectorAD &state, VectorAD &input) {
          return static_cast<Model*>(this)->Dynamics(state, input);
        }

        VectorAD DynamicsInterface(VectorAD &state, VectorAD &input, VectorAD &params) {
          return static_cast<Model*>(this)->Dynamics(state, input, params);
        }

        VectorAD Dynamics(VectorAD &state, VectorAD &input, VectorAD &params) {
          return static_cast<Model*>(this)->Dynamics(state, input);
        }


        // discretization with Runge-Kutta 4th order
        // ns steps over the sampling time interval dt_
        VectorAD step(VectorAD &state, VectorAD &input) {

          VectorAD state_next = state;

          int ns = 1;
          double h = dt_ / ns;

          for (int i = 0; i < ns; ++i) {
            VectorAD k1 = DynamicsInterface(state, input);

            VectorAD temp1 = state + 0.5 * k1 * h;
            VectorAD k2 = DynamicsInterface(temp1, input);

            VectorAD temp2 = state + 0.5 * k2 * h;
            VectorAD k3 = DynamicsInterface(temp2, input);

            VectorAD temp3 = state + k3 * h;
            VectorAD k4 = DynamicsInterface(temp3, input);

            state_next += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
          }

          return state_next;
        }

        VectorAD step(VectorAD &state, VectorAD &input, VectorAD &params) {

          VectorAD state_next = state;

          int ns = 1;
          double h = dt_ / ns;

          for (int i = 0; i < ns; ++i) {
            VectorAD k1 = DynamicsInterface(state, input, params);

            VectorAD temp1 = state + 0.5 * k1 * h;
            VectorAD k2 = DynamicsInterface(temp1, input, params);

            VectorAD temp2 = state + 0.5 * k2 * h;
            VectorAD k3 = DynamicsInterface(temp2, input, params);

            VectorAD temp3 = state + k3 * h;
            VectorAD k4 = DynamicsInterface(temp3, input, params);

            state_next += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
          }

          return state_next;
        }


        Vector integrate(const Vector & _state,const Vector & _input) {
          return static_cast<Model*>(this)->integrator(_state, _input);
        }

        virtual // Integrator interface for plain Eigen vectors
        Vector integrator(const Vector & _state,const Vector & _input) {
          VectorAD state = _state;
          VectorAD input = _input;

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
        Matrix dFdp(VectorAD &state, VectorAD &input, VectorAD &params) {
          return autodiff::jacobian([&](VectorAD &state, VectorAD &input, VectorAD &params) {return step(state,input,params); },
                                    wrt(params), at(state, input, params));
        }

        MatrixAD dFdxu(VectorAD &state, VectorAD &input, VectorAD &params) {
          MatrixAD res(nx_, nx_ + nu_);
          res.leftCols(nx_) = Ad(state, input);
          res.rightCols(nu_) = Bd(state, input);

          return res;
        }

/*        virtual MatrixAD dFdxup(VectorAD &state, VectorAD &input, VectorAD &params) {
          return dFdxu(state, input, params);
        }*/
        //------------------------------------------------------------------------------------------------------------------

        // constraints (to be implemented) and their sensitivities

        VectorAD ConstraintInterface(VectorAD &state, VectorAD &input, int t_idx) {
          return static_cast<Model*>(this)->Constraint(state, input, t_idx);
        }

        VectorAD EndConstraintInterface(VectorAD &state, int t_idx) {
          return static_cast<Model*>(this)->EndConstraint(state, t_idx);
        }

        Matrix Cd(VectorAD &state, VectorAD &input, const int t_idx) {
          return autodiff::jacobian([&](VectorAD &state, VectorAD &input, const int t_idx){
            return ConstraintInterface(state, input, t_idx); },
                                    wrt(state),at(state, input, t_idx),h_eval_);
        }

        Matrix Dd(VectorAD &state, VectorAD &input, const int t_idx) {
          return autodiff::jacobian([&](VectorAD &state, VectorAD &input, const int t_idx) {
            return ConstraintInterface(state, input, t_idx); },
                                    wrt(input),
                                    at(state, input, t_idx),
                                    h_eval_);
        }

/*        virtual MatrixAD dConstrdxu(VectorAD &state, VectorAD &input, VectorAD &params, const int t_idx) {
          MatrixAD res(nh_, nx_+nu_);
          res.leftCols(nx_) = Cd(state, input, t_idx);
          res.rightCols(nu_) = Dd(state, input, t_idx);

          return res;
        }*/


        Matrix Cd_e(VectorAD &state, const int t_idx) {
          return autodiff::jacobian([&](VectorAD &state, const int t_idx) { return EndConstraintInterface(state, t_idx); },
                                    wrt(state),
                                    at(state, t_idx),
                                    he_eval_);
        }
        //------------------------------------------------------------------------------------------------------------------

        Vector GetModelParams() {
          return static_cast<Model*>(this)->GetModelParams();
        }



    };

}