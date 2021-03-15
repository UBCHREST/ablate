#ifndef ABLATELIBRARY_TRACER_HPP
#define ABLATELIBRARY_TRACER_HPP
#include "particles.hpp"

namespace ablate::particles {
class Tracer : public Particles {
   public:
    Tracer(std::string name, int ndims, std::map<std::string, std::string> arguments, std::shared_ptr<particles::initializers::Initializer> initializer,
           std::shared_ptr<mathFunctions::MathFunction> exactSolution);
    ~Tracer() override;

    void InitializeFlow(std::shared_ptr<flow::Flow> flow, std::shared_ptr<solve::TimeStepper> flowTimeStepper) override;
};
}  // namespace ablate::particles

#endif  // ABLATELIBRARY_TRACER_HPP
