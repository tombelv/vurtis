//
// Created by tom on 21/03/23.
//

#ifndef EXAMPLE_COST_FUNCTION_H
#define EXAMPLE_COST_FUNCTION_H

#include "vurtis/vurtis.h"


class Cost : public vurtis::CostBase {
public:
    explicit Cost(const vurtis::Vector& x_ref, const vurtis::Vector& u_ref) : vurtis::CostBase(x_ref, u_ref) {}

    // function LeastSquareCost: R(x) such that CostFunctionPointwise=0.5*R(x)'*R(x)
    vurtis::VectorAD LeastSquareCost(vurtis::VectorAD &state, vurtis::VectorAD &input, const vurtis::Vector &state_ref, const vurtis::Vector &input_ref) {

      vurtis::VectorAD cost(5);

      cost[0] = 0.0 * (state[0]-state_ref[0]);
      cost[1] = 0.0 * (state[1]-state_ref[1]);
      cost[2] = 0.0 * (state[2]-state_ref[2]);
      cost[3] = 1e-2 * (input[0]);
      cost[4] = 1e-2 * (input[1]);
      return cost;
    }

    vurtis::VectorAD LeastSquareCostTerminal(vurtis::VectorAD &state, const vurtis::Vector &state_ref) {

      return We*(state-state_ref);
    }

private:

    const vurtis::Matrix We = Eigen::Vector3d(1,1,0.0).asDiagonal();
};

#endif //EXAMPLE_COST_FUNCTION_H
