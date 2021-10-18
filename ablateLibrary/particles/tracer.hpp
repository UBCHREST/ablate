#ifndef ABLATELIBRARY_TRACER_HPP
#define ABLATELIBRARY_TRACER_HPP
#include "particles.hpp"

namespace ablate::particles {
class Tracer : public Particles {
   public:
    Tracer(std::string solverId, std::string region,std::shared_ptr<parameters::Parameters> options, int ndims, std::shared_ptr<particles::initializers::Initializer> initializer, std::shared_ptr<mathFunctions::MathFunction> exactSolution);
    ~Tracer() override;

    void Initialize() override;

   private:
    static PetscErrorCode freeStreaming(TS ts, PetscReal t, Vec X, Vec F, void* ctx);
};
}  // namespace ablate::particles

#endif  // ABLATELIBRARY_TRACER_HPP
