#ifndef ABLATELIBRARY_IGNITIONDELAYTEMPERATURE_HPP
#define ABLATELIBRARY_IGNITIONDELAYTEMPERATURE_HPP

#include <monitors/logs/log.hpp>
#include "eos/eos.hpp"
#include "monitor.hpp"

namespace ablate::monitors {

/**
 * The ignition delay monitor logs the temperature computes the ignition delay from it.
 */
class IgnitionDelayTemperature : public Monitor {
    static PetscErrorCode MonitorIgnition(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx);

    // store the eos inorder to compute temperature
    const std::shared_ptr<eos::EOS> eos;
    const double thresholdTemperature;
    const std::shared_ptr<logs::Log> log;
    const std::shared_ptr<logs::Log> historyLog;

    const std::string species;
    const std::vector<double> location;

    // The offset for the euler and mass fractions in the solution array
    PetscInt eulerId;
    PetscInt yiId;
    PetscInt cellOfInterest;

    std::vector<double> timeHistory;
    std::vector<double> temperatureHistory;

   public:
    explicit IgnitionDelayTemperature(std::shared_ptr<eos::EOS>, std::vector<double> location, double thresholdTemperature, std::shared_ptr<logs::Log> log = {},
                                      std::shared_ptr<logs::Log> historyLog = {});
    ~IgnitionDelayTemperature() override;

    void Register(std::shared_ptr<solver::Solver>) override;
    PetscMonitorFunction GetPetscFunction() override { return MonitorIgnition; }
};
}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_IGNITIONDELAYTEMPERATURE_HPP
