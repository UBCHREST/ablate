#ifndef ABLATELIBRARY_FINITEVOLUME_CHEMISTRY_HPP
#define ABLATELIBRARY_FINITEVOLUME_CHEMISTRY_HPP

#include <memory>
#include "eos/chemistryModel.hpp"
#include "process.hpp"

namespace ablate::finiteVolume::processes {

class Chemistry : public Process, public ablate::utilities::Loggable<Chemistry> {
   private:
    //! store the eos that will be used to create the calculator
    const std::shared_ptr<ablate::eos::ChemistryModel> chemistryModel;

    //! the current active chemistry calculator
    std::shared_ptr<ablate::eos::ChemistryModel::SourceCalculator> sourceCalculator;

    /**
     * private function to compute the energy and densityYi source terms over the next dt
     * @param flowTs
     * @param flow
     * @return
     */
    PetscErrorCode ChemistryPreStage(TS flowTs, ablate::solver::Solver &flow, PetscReal stagetime);

    /**
     * static function to add chemistry source terms
     * @param solver
     * @param dm
     * @param time
     * @param locX
     * @param fVec
     * @param ctx
     * @return
     */
    static PetscErrorCode AddChemistrySourceToFlow(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec fVec, void *ctx);

   public:
    /**
     * The chemistry processes need a chemistry model
     */
    explicit Chemistry(std::shared_ptr<ablate::eos::ChemistryModel>);

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

    /**
     * compute/setup memory for the current mesh
     * @param flow
     */
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

    /**
     * public function to copy the source terms to a locFVec
     * @param solver
     * @param fVec
     */
    void AddChemistrySourceToFlow(const FiniteVolumeSolver &solver, Vec locX, Vec locFVec);
};
}  // namespace ablate::finiteVolume::processes
#endif
