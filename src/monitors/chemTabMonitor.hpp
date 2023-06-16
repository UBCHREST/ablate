#ifndef ABLATELIBRARY_CHEMTABNMONITOR_HPP
#define ABLATELIBRARY_CHEMTABNMONITOR_HPP

#include <memory>
#include "eos/chemTab.hpp"
#include "fieldMonitor.hpp"
#include "finiteVolume/processes/chemistry.hpp"
#include "mixtureFractionCalculator.hpp"

/**
 * This class reports the output values for chemTab
 */
namespace ablate::monitors {

class ChemTabMonitor : public FieldMonitor {
   private:
    //! the base chemTab
    const std::shared_ptr<eos::ChemTab> chemTab;

    //! store a reference to a function to compute density from solution field
    eos::ThermodynamicFunction densityFunction;

    //! store an optional pointer to the TChemReactions to output chemistry source terms
    std::shared_ptr<ablate::finiteVolume::processes::Chemistry> chemistry;

   public:
    /**
     * Create the chemTab monitor with chemTab
     */
    explicit ChemTabMonitor(const std::shared_ptr<ablate::eos::ChemistryModel>& chemTab);

    /**
     * Update the fields and save the results to an hdf5 file
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    PetscErrorCode Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;

    /**
     * this call setups the monitor and defines the mixture fraction fields
     * method
     * @param solverIn
     */
    void Register(std::shared_ptr<solver::Solver> solverIn) override;
};

}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_MIXTUREFRACTIONMONITOR_HPP
