#include "eulerianSourceAccessor.hpp"

#include <utility>
#include "particles/particleSolver.hpp"

ablate::particles::accessors::EulerianSourceAccessor::EulerianSourceAccessor(bool cachePointData, std::shared_ptr<ablate::domain::SubDomain> subDomain, const DM& swarmDm)
    : Accessor(cachePointData), subDomain(std::move(subDomain)), swarmDm(swarmDm) {}

ablate::particles::accessors::PointData ablate::particles::accessors::EulerianSourceAccessor::CreateData(const std::string& fieldName) {
    // get the source field from the swarm dm
    PetscScalar* values;
    PetscInt dataSize;  // The data size might be bigger than the number of components because the vector holds all the source terms
    DMSwarmGetField(swarmDm, CoupledSourceTerm, &dataSize, nullptr, (void**)&values) >> utilities::PetscUtilities::checkError;

    // Register the cleanup
    RegisterCleanupFunction([=]() { DMSwarmRestoreField(swarmDm, CoupledSourceTerm, nullptr, nullptr, (void**)&values) >> utilities::PetscUtilities::checkError; });

    // get the solution field from the subDomain to determine the number of components
    const auto& eulerianField = subDomain->GetField(fieldName);
    // the offset is assumed to be the same offset for both the eulerian and lagrangian implementations
    return {values, eulerianField.numberComponents, dataSize, eulerianField.offset};
}
