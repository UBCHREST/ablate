#ifndef ABLATELIBRARY_BUOYANCY_HPP
#define ABLATELIBRARY_BUOYANCY_HPP

#include "flowProcess.hpp"

namespace ablate::finiteVolume::processes {

class Buoyancy : public FlowProcess {
   private:
    const std::vector<PetscReal> buoyancyVector;
    /**
     * Compute and store the avg density in the domain
     */
    PetscReal densityAvg = NAN;

    /**
     * private function to compute the average density in the domain
     * @param flowTs
     * @param flow
     * @return
     */
    PetscErrorCode UpdateAverageDensity(TS flowTs, ablate::solver::Solver& flow);

    /**
     * private function to compute gravity source
     * @return
     */
    static PetscErrorCode ComputeBuoyancySource(PetscInt dim, PetscReal time, const PetscFVCellGeom* cg, const PetscInt uOff[], const PetscScalar u[], const PetscInt aOff[], const PetscScalar a[],
                                                PetscScalar f[], void* ctx);

   public:
    explicit Buoyancy(std::vector<double> buoyancyVector);
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver& fv) override;
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_BUOYANCY_HPP
