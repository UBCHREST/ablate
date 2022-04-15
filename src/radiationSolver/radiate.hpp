#ifndef ABLATELIBRARY_radiate_HPP
#define ABLATELIBRARY_radiate_HPP

#include "radiationSolver.hpp"
#include "radiationSolver/radiationProcess.hpp"
#include "finiteVolume/processes/eulerTransport.hpp"

namespace ablate::radiationSolver {

class radiate : public RadiationProcess {
   public:
    explicit radiate();

    void Initialize(ablate::radiationSolver::RadiationSolver& bSolver) override;

    static PetscErrorCode UpdateAuxTemperatureField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[],
                                         const PetscScalar* conservedValues, PetscScalar* auxField, void* ctx);

    //std::vector<std::string> fieldNames;

   private:
    ablate::finiteVolume::processes::EulerTransport::UpdateTemperatureData updateTemperatureData{};
};

}  // namespace ablate::radiationSolver::lodi
#endif  // ABLATELIBRARY_radiate_HPP
