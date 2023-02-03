#ifndef ABLATELIBRARY_RADIATIONFIELDMONITOR_H
#define ABLATELIBRARY_RADIATIONFIELDMONITOR_H

#include "eos/eos.hpp"
#include "eos/radiationProperties/radiationProperties.hpp"
#include "fieldMonitor.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "io/interval/fixedInterval.hpp"
#include "utilities/constants.hpp"

namespace ablate::monitors {

class RadiationFieldMonitor : public FieldMonitor {
    enum FieldPlacements { intensity, absorption, fieldsStart };
    using tp = ablate::eos::ThermodynamicProperty;
    using fLoc = ablate::domain::FieldLocation;

   public:
    RadiationFieldMonitor(const std::shared_ptr<ablate::eos::EOS> eosIn, std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn,
                          std::shared_ptr<io::interval::Interval> intervalIn);

    void Register(std::shared_ptr<ablate::solver::Solver> solverIn) override;
    void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;
    static PetscErrorCode MonitorRadiation(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx);
    PetscMonitorFunction GetPetscFunction() override { return MonitorRadiation; }

   private:
    //! model used to provided the absorptivity function
    const std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModel;  //! Radiation properties model from which the absorption is calculated.

    //! hold a pointer to the absorptivity function
    eos::ThermodynamicTemperatureFunction absorptivityFunction;  //! Absorption function for the radiation properties. This comes from the model.

    const std::shared_ptr<ablate::eos::EOS> eos;             //! Equation of state used for computing flow field properties.
    const std::shared_ptr<io::interval::Interval> interval;  //! Interval of the radiation properties recording.
    PetscInt step;

    // Create suffix vector
    std::vector<std::string> fieldNames{"radiationIntensity", "absorptionCoefficient"};
};
}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_RADIATIONFIELDMONITOR_H
