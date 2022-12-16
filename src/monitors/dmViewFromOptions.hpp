#ifndef ABLATELIBRARY_DMVIEWFROMOPTIONS_HPP
#define ABLATELIBRARY_DMVIEWFROMOPTIONS_HPP

#include <petsc.h>
#include <parameters/parameters.hpp>
#include <string>
#include "monitor.hpp"

namespace ablate::monitors {
/**
 * This class replicates the functionality provided by PetscObjectViewFromOptions
 *
 * If the options string is passed, an options object is created and set with the options
 * If not set, options is assumed to the global values and the optionName should be used
 */
class DmViewFromOptions : public Monitor, public ablate::domain::modifiers::Modifier {
   public:
    enum class Scope { INITIAL, MONITOR };

   private:
    PetscOptions petscOptions;
    const std::string optionName;
    const Scope scope;

    static PetscErrorCode CallDmViewFromOptions(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx);

    PetscErrorCode DMViewFromOptions(DM dm);

   public:
    /**
     * public constructor allow this monitor to set scope
     * @param scope
     * @param options
     * @param optionName
     */
    explicit DmViewFromOptions(Scope scope, std::string options = {}, const std::string& optionName = {});

    /**
     * constructor without allowing the monitor to be set (Set to INITIAL by default)
     * @param options
     * @param optionName
     */
    explicit DmViewFromOptions(std::string options = {}, std::string optionName = {});
    ~DmViewFromOptions() override;

    void Register(std::shared_ptr<solver::Solver>) override;
    PetscMonitorFunction GetPetscFunction() override { return CallDmViewFromOptions; }

    /**
     * Return the name of this modifier
     * @return
     */
    std::string ToString() const override { return "ablate::monitors::DmViewFromOptions"; }

    /**
     * Allow this monitor to also be used as modifier to save results
     */
    void Modify(DM&) override;
};

/**
 * Support function for the Scope Enum
 * @param os
 * @param v
 * @return
 */
std::ostream& operator<<(std::ostream& os, const DmViewFromOptions::Scope& v);
/**
 * Support function for the Scope Enum
 * @param os
 * @param v
 * @return
 */
std::istream& operator>>(std::istream& is, DmViewFromOptions::Scope& v);
}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_DMVIEWFROMOPTIONS_HPP
