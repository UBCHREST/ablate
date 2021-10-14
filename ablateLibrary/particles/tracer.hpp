#ifndef ABLATELIBRARY_TRACER_HPP
#define ABLATELIBRARY_TRACER_HPP
#include "particles.hpp"

namespace ablate::particles {
class Tracer : public Particles {
   public:
    Tracer(std::string name, int ndims, std::shared_ptr<particles::initializers::Initializer> initializer, std::shared_ptr<mathFunctions::MathFunction> exactSolution = {},
           std::shared_ptr<parameters::Parameters> options = {});
    ~Tracer() override;

    void Initialize(std::shared_ptr<domain::SubDomain> flow) override;

   private:
    static PetscErrorCode freeStreaming(TS ts, PetscReal t, Vec X, Vec F, void* ctx);
};
}  // namespace ablate::particles

#endif  // ABLATELIBRARY_TRACER_HPP
