#ifndef ABLATELIBRARY_BOUYANCY_HPP
#define ABLATELIBRARY_BOUYANCY_HPP

#include "flowProcess.hpp"

namespace ablate::finiteVolume::processes {

class Bouyancy : public FlowProcess {
   private:
    const std::vector<PetscReal> bouyancyVector;
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
    static PetscErrorCode ComputeBouyancySource(PetscInt dim, PetscReal time, const PetscFVCellGeom* cg, const PetscInt uOff[], const PetscScalar u[], const PetscScalar* const gradU[],
                                               const PetscInt aOff[], const PetscScalar a[], const PetscScalar* const gradA[], PetscScalar f[], void* ctx);

   public:
    explicit Bouyancy(std::vector<double> bouyancyVector);
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver& fv) override;
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_BOUYANCY_HPP
