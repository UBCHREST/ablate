#ifndef ABLATELIBRARY_HDF5OUTPUTFLOW_HPP
#define ABLATELIBRARY_HDF5OUTPUTFLOW_HPP
#include <filesystem>
#include "monitors/particles/particleMonitor.hpp"
#include "monitors/hdf5Output.hpp"
namespace ablate::monitors::particles{
class Hdf5OutputParticle: public monitors::particles::ParticleMonitor, monitors::Hdf5Output{
   private:
    std::shared_ptr<ablate::particles::Particles> particles = nullptr;
    static PetscErrorCode OutputParticles(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx);

   public:
    Hdf5OutputParticle() = default;
    ~Hdf5OutputParticle() override = default;

    void Register(std::shared_ptr<ablate::particles::Particles>) override;

    PetscMonitorFunction GetPetscFunction() override{
        return OutputParticles;
    }
};
}


#endif  // ABLATELIBRARY_HDF5OUTPUTFLOW_HPP
