#ifndef ABLATELIBRARY_PARTICLES_HPP
#define ABLATELIBRARY_PARTICLES_HPP

#include <memory>
#include "flow/flow.hpp"
#include "mathFunctions/mathFunction.hpp"
#include "particles.h"
#include "particles/initializers/initializer.hpp"
#include "solve/timeStepper.hpp"

namespace ablate::particles {

class Particles {
   private:
    std::shared_ptr<mathFunctions::MathFunction> exactSolution = nullptr;
    std::shared_ptr<particles::initializers::Initializer> initializer = nullptr;

   protected:
    const std::string name;
    ParticleData particleData;
    TS particleTs;

    void SetExactSolution(std::shared_ptr<mathFunctions::MathFunction> exactSolution);

   public:
    explicit Particles(std::string name, std::map<std::string, std::string> arguments, std::shared_ptr<particles::initializers::Initializer> initializer);
    virtual ~Particles() = default;

    const std::string& GetName() const { return name; }

    ParticleData& GetParticleData() { return particleData; }

    virtual void InitializeFlow(std::shared_ptr<flow::Flow> flow, std::shared_ptr<solve::TimeStepper> flowTimeStepper);
};
}  // namespace ablate::particles
#endif  // ABLATELIBRARY_PARTICLES_HPP
