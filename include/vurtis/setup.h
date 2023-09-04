#pragma once

#include "utils.h"

namespace vurtis {


    template <typename Setup>
    struct SetupBase {
        static constexpr double dt = Setup::dt;
        static constexpr int N = Setup::N;
        static constexpr int nx = Setup::nx;
        static constexpr int nu = Setup::nu;
        static constexpr int nz = Setup::nz;
        static constexpr int nz_e = Setup::nz_e;
        static constexpr int nh = Setup::nh;
        static constexpr int nh_e = Setup::nh_e;
        static constexpr int np = Setup::np;

        static constexpr int n_dec_var = nx*(N+1) + nu*N;
        static constexpr int n_eq_constr = nx*(N+1);
        static constexpr int n_ineq_constr = nh*N + nh_e;
    };

}
