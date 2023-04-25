#ifndef UNICYCLE_SENSITIVITY_SETUP_H
#define UNICYCLE_SENSITIVITY_SETUP_H

#include "utils.h"
#include "toml.h"

namespace vurtis {

    class ProblemSetup {

    public:

        explicit ProblemSetup(std::string config_path) {

          toml::table config = toml::parse_file(config_path);

          dT = config.at_path("dT").as_floating_point()->get();

          nx = config.at_path("nx").as_integer()->get();
          nu = config.at_path("nu").as_integer()->get();
          nz = config.at_path("nz").as_integer()->get();
          nh = config.at_path("nh").as_integer()->get();
          nh_e = config.at_path("nh_e").as_integer()->get();
          N = config.at_path("N").as_integer()->get();


          num_parameters = config.at_path("num_parameters").as_integer()->get();

          x0.resize(nx);
          for(int i=0; i<nx; ++i) x0(i) = config.at_path("x0["+std::to_string(i)+"]").as_floating_point()->get();

          nlp_parameters = Matrix::Zero(num_parameters, N+1);

        }

        double dT;

        int nx; // number of states of dynamic model
        int nu; // number of inputs
        int nz; // dimension of the least square cost function
        int nh; // number of constraints (step=0,...,N-1)
        int nh_e; // number of end-constraints (step=N)
        int N; // number of steps
        int num_parameters; // number of nlp parameters

        Matrix nlp_parameters; // initialization of parameters
        Vector x0; // initial state



    };


}

#endif //UNICYCLE_SENSITIVITY_SETUP_H
