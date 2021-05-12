#ifndef CMDSTAN_ARGUMENTS_ARG_LOG_PROB_GRAD_HPP
#define CMDSTAN_ARGUMENTS_ARG_LOG_PROB_GRAD_HPP

#include <cmdstan/arguments/arg_generate_quantities_fitted_params.hpp>
#include <cmdstan/arguments/categorical_argument.hpp>

namespace cmdstan {

class arg_log_prob_grad : public categorical_argument {
 public:
  arg_log_prob_grad() {
    _name = "log_prob_grad";
    _description = "Generate lp__ and gradient";

    _subarguments.push_back(new arg_generate_quantities_fitted_params());
  }
};

}  // namespace cmdstan
#endif
