#ifndef QUADROTOR_SENSITIVITY_PARAMETERS_BASE_H
#define QUADROTOR_SENSITIVITY_PARAMETERS_BASE_H

namespace vurtis{

    struct ParametersBase {
        virtual void init() = 0;
    };
}



#endif //QUADROTOR_SENSITIVITY_PARAMETERS_BASE_H
