#ifndef ABLATELIBRARY_RADIATIONFIELDMONITOR_H
#define ABLATELIBRARY_RADIATIONFIELDMONITOR_H

#include "fieldMonitor.hpp"
#include "eos/eos.hpp"
#include "io/interval/fixedInterval.hpp"
#include "eos/radiationProperties/radiationProperties.hpp"

namespace ablate::monitors {

    class RadiationFieldMonitor : public FieldMonitor  {

        enum FieldPlacements { densitySum, densityDtSum, fieldsStart };
        using tp = ablate::eos::ThermodynamicProperty;
        using fLoc = ablate::domain::FieldLocation;

       public:
        RadiationFieldMonitor(const std::shared_ptr<ablate::eos::EOS> eosIn, std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<io::interval::Interval> intervalIn);

        ~RadiationFieldMonitor();

        void Register(std::shared_ptr<ablate::solver::Solver> solverIn) override;
        void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;

       private:
        //! model used to provided the absorptivity function
        const std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModel;

        //! hold a pointer to the absorptivity function
        eos::ThermodynamicTemperatureFunction absorptivityFunction;

        const std::shared_ptr<ablate::eos::EOS> eos;
        const std::shared_ptr<io::interval::Interval> interval;

    };
}
#endif  // ABLATELIBRARY_RADIATIONFIELDMONITOR_H
