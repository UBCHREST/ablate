#ifndef ABLATELIBRARY_VOLUMERADIATION_HPP
#define ABLATELIBRARY_VOLUMERADIATION_HPP

#include "radiation.hpp"
#include "io/interval/interval.hpp"

namespace ablate::radiation {

class VolumeRadiation : public Radiation, public solver::CellSolver, public solver::RHSFunction {
   public:
    /**
     * Function passed into PETSc to compute the FV RHS
     * @param dm
     * @param time
     * @param locXVec
     * @param globFVec
     * @param ctx
     * @return
     */
    PetscErrorCode ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec) override;

    void Initialize() override;
    void Setup() override;
    void Register(std::shared_ptr<ablate::domain::SubDomain> subDomain) override;

    /**
     *
     * @param solverId the id for this solver
     * @param region the boundary cell region
     * @param rayNumber
     * @param options other options
     */
    VolumeRadiation(const std::string& solverId1, const std::shared_ptr<domain::Region>& region, std::shared_ptr<domain::Region> fieldBoundary, const PetscInt raynumber, std::shared_ptr<io::interval::Interval> interval,
                    const std::shared_ptr<parameters::Parameters>& options1, std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<monitors::logs::Log> unnamed1);

    ~VolumeRadiation();

    PetscErrorCode RadiationPreStep(TS ts, ablate::solver::Solver& solver);

    const std::shared_ptr<io::interval::Interval> interval;
};
}  // namespace ablate::radiation
#endif  // ABLATELIBRARY_VOLUMERADIATION_HPP
