#ifndef ABLATELIBRARY_CHEMTABMONITOR_HPP
#define ABLATELIBRARY_CHEMTABMONITOR_HPP

#include "eos/chemTab.hpp"
#include "monitor.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"

namespace ablate::monitors {
/**
 * The chemTab monitor is used to add decoded mass fractions to the aux vector and compute the output
 */
class ChemTabMonitor : public Monitor {
   private:
    // hold the chemTab monitor to decode the progress variable
    std::shared_ptr<ablate::eos::ChemTab> chemTabModel = nullptr;

    // hold the density function to decode progress variable
    eos::ThermodynamicFunction densityFunction;

    /**
     * Function decodes the mass fractions from chemTab to update the aux field
     * @param ts
     * @param time
     * @param initialStage
     * @param locX
     * @param ctx
     * @return
     */
    static PetscErrorCode DecodeMassFractions(ablate::finiteVolume::FiniteVolumeSolver&, TS ts, PetscReal time, bool initialStage, Vec locX, void* ctx);

   public:
    /**
     * Override this function to setup the monitor
     * @param solverIn
     */
    explicit ChemTabMonitor(const std::shared_ptr<ablate::eos::EOS>& eos);

    /**
     * Override this function to setup the monitor and add a pre rhs step
     * @param solverIn
     */
    void Register(std::shared_ptr<solver::Solver> solverIn) override;

    /**
     * Helper class to setup the required fields for the monitor
     */
    class Fields : public domain::FieldDescriptor {
        // hold the chemTab monitor to decode the progress variable
        std::shared_ptr<ablate::eos::ChemTab> chemTabModel = nullptr;

        // the region to define the yi field and compute the decoded yi
        const std::shared_ptr<domain::Region> region;

       public:
        /**
         * Override this function to setup the monitor
         * @param solverIn
         */
        explicit Fields(const std::shared_ptr<ablate::eos::EOS>& eos, std::shared_ptr<domain::Region> region = nullptr);

        /**
         * Return the field need for the monitor
         * @return
         */
        std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;
    };
};

}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_CHEMTABMONITOR_HPP
