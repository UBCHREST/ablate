#ifndef ABLATELIBRARY_VOLUMERADIATION_HPP
#define ABLATELIBRARY_VOLUMERADIATION_HPP

#include "domain/dynamicRange.hpp"
#include "io/interval/interval.hpp"
#include "radiation.hpp"

namespace ablate::radiation {

class VolumeRadiation : public solver::CellSolver, public solver::RHSFunction {
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
    VolumeRadiation(const std::string& solverId1, const std::shared_ptr<domain::Region>& region, const std::shared_ptr<io::interval::Interval>& interval,
                    std::shared_ptr<radiation::Radiation> radiation, const std::shared_ptr<parameters::Parameters>& options1, const std::shared_ptr<monitors::logs::Log>& unnamed1);

    ~VolumeRadiation();

    /**
     * serves to update the radiation
     * @param time
     * @param locX
     * @return
     */
    PetscErrorCode PreRHSFunction(TS ts, PetscReal time, bool initialStage, Vec locX) override;

   private:
    const std::shared_ptr<io::interval::Interval> interval;
    std::shared_ptr<ablate::radiation::Radiation> radiation;
    ablate::domain::DynamicRange radiationCellRange;

    //! hold a pointer to the absorptivity function
    eos::ThermodynamicTemperatureFunction absorptivityFunction;
    eos::ThermodynamicTemperatureFunction emissivityFunction;
};
}  // namespace ablate::radiation
#endif  // ABLATELIBRARY_VOLUMERADIATION_HPP
