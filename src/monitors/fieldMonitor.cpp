#include "fieldMonitor.hpp"
#include "domain/dmTransfer.hpp"

void ablate::monitors::FieldMonitor::Register(std::string id, std::shared_ptr<solver::Solver> solverIn, std::vector<std::shared_ptr<domain::FieldDescriptor>> fieldDescriptors) {
    // Set the local variables
    ablate::monitors::Monitor::Register(solverIn);

    // make sure that each of the fields is for the entire domain
    for (const auto& fieldDescriptor : fieldDescriptors) {
        for (const auto& field : fieldDescriptor->GetFields()) {
            if (field->region != domain::Region::ENTIREDOMAIN) {
                throw std::invalid_argument("The ablate::monitors::FieldMonito requires all fields to be defined over the entire domain");
            }
        }
    }

    // Create a subDomain only over this solver region
    DM subDm;
    solverIn->GetSubDomain().CreateEmptySubDM(&subDm, solverIn->GetRegion());

    // name the subDm
    PetscObjectSetName((PetscObject)subDm, id.c_str()) >> checkError;

    // Create a domain
    monitorDomain = std::make_shared<ablate::domain::DMTransfer>(subDm, fieldDescriptors);

    // Init the monitorDomain
    monitorDomain->InitializeSubDomains({}, {});

    // Get the subdomain for this domain.  There should only be one ds created
    monitorSubDomain = monitorDomain->GetSubDomain(nullptr);

    // remove the name for this vector
    PetscObjectSetName((PetscObject)monitorSubDomain->GetSolutionVector(), "monitor") >> checkError;
}

void ablate::monitors::FieldMonitor::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;
    monitorSubDomain->Save(viewer, sequenceNumber, time);
    PetscFunctionReturnVoid();
}
void ablate::monitors::FieldMonitor::Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) { monitorSubDomain->Restore(viewer, sequenceNumber, time); }
