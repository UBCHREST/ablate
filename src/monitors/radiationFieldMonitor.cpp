#include "radiationFieldMonitor.hpp"

ablate::monitors::RadiationFieldMonitor::RadiationFieldMonitor(const std::shared_ptr<ablate::eos::EOS> eosIn, std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn,
                                                               std::shared_ptr<io::interval::Interval> intervalIn)
    : eos(eosIn), interval(intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>()) {}

void ablate::monitors::RadiationFieldMonitor::Register(std::shared_ptr<ablate::solver::Solver> solverIn) {
    Monitor::Register(solverIn);

    // Create the monitor name
    std::string dmID = "radiationFieldMonitor";

    // Create suffix vector
    std::vector<std::string> fieldNames{"radiationIntensity", "absorptionCoefficient"};

    std::vector<std::shared_ptr<domain::FieldDescriptor>> fields(fieldNames.size() + FieldPlacements::fieldsStart, nullptr);

    for (std::size_t f = 0; f < fieldNames.size(); f++) {
        fields[FieldPlacements::fieldsStart + f] = std::make_shared<domain::FieldDescription>(fieldNames[f], fieldNames[f], processedCompNames[f], domain::FieldLocation::SOL, domain::FieldType::FVM);
    }

    // Register all fields with the monitorDomain
    ablate::monitors::FieldMonitor::Register(dmID, solverIn, fields);

    // Get the density thermodynamic function
    absorptivityFunction = radiationModel->GetRadiationPropertiesTemperatureFunction(eos::radiationProperties::RadiationProperty::Absorptivity, solverIn->GetSubDomain().GetFields());
}

void ablate::monitors::RadiationFieldMonitor::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    // Perform the principal save
    ablate::monitors::FieldMonitor::Save(viewer, sequenceNumber, time);

    // Save the step number
    //    ablate::io::Serializable::SaveKeyValue(viewer, "step", step);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::RadiationFieldMonitor, "A solver for radiative heat transfer in participating media",
         ARG(ablate::eos::radiationProperties::RadiationModel, "radiationProperties", "properties model for the output of radiation properties within the field"),
         OPT(ablate::io::interval::Interval, "interval", "The monitor output interval"));