#ifndef ABLATELIBRARY_IGNITIONDELAYPEAKYI_HPP
#define ABLATELIBRARY_IGNITIONDELAYPEAKYI_HPP

#include <monitors/logs/log.hpp>
#include "monitor.hpp"

namespace ablate::monitors {

/**
 * The ignition delay monitor logs the mass fraction of the specified species and computes the ignition delay from its peak value.
 */
class IgnitionDelayPeakYi : public Monitor {
   private:
    static PetscErrorCode MonitorIgnition(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx);
    const std::shared_ptr<logs::Log> log;
    const std::shared_ptr<logs::Log> historyLog;

    const std::string species;
    const std::vector<double> location;

    // The offset for the density and mass fractions in the solution array
    PetscInt eulerId;
    PetscInt yiId;
    PetscInt yiOffset;
    PetscInt cellOfInterest;

    std::vector<double> timeHistory;
    std::vector<double> yiHistory;

   public:
    explicit IgnitionDelayPeakYi(std::string species, std::vector<double> location, std::shared_ptr<logs::Log> log = {}, std::shared_ptr<logs::Log> historyLogIn = {});
    ~IgnitionDelayPeakYi() override;

    void Register(std::shared_ptr<solver::Solver>) override;
    PetscMonitorFunction GetPetscFunction() override { return MonitorIgnition; }
};

}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_IGNITIONDELAYPEAKYI_HPP
