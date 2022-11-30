#ifndef ABLATELIBRARY_MIXTUREFRACTIONMONITOR_HPP
#define ABLATELIBRARY_MIXTUREFRACTIONMONITOR_HPP

#include <memory>
#include "mixtureFractionCalculator.hpp"
#include "fieldMonitor.hpp"
#include "finiteVolume/processes/tChemReactions.hpp"

/**
 * This class computes the mixture fraction for each point in the domain and outputs zMix, Yi, and source terms to the hdf5 file
 */
namespace ablate::monitors {

class MixtureFractionMonitor : public FieldMonitor {
   private:
    //! the mixture fraction calculator
    const std::shared_ptr<MixtureFractionCalculator> mixtureFractionCalculator;

    //! store a reference to a function to compute density from solution field
    eos::ThermodynamicFunction densityFunction;

    //! store an optional pointer to the TChemReactions to output chemistry source terms
    std::shared_ptr<ablate::finiteVolume::processes::TChemReactions> tChemReactions;

   public:
    /**
     * Create the mixture fraction monitor using a mixture fraction calculator
     */
    explicit MixtureFractionMonitor(std::shared_ptr<MixtureFractionCalculator>);

    /**
     * Update the fields and save the results to an hdf5 file
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;

    /**
     * this call setups the monitor and defines the mixture fraction fields
     * method
     * @param solverIn
     */
    void Register(std::shared_ptr<solver::Solver> solverIn) override;
};

}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_MIXTUREFRACTIONMONITOR_HPP
