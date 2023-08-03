#include "variableChange.hpp"

#include <domain/subDomain.hpp>
#include <utility>

ablate::solver::criteria::VariableChange::VariableChange(std::string variableName, double convergenceTolerance, ablate::utilities::MathUtilities::Norm convergenceNorm,
                                                         std::shared_ptr<ablate::domain::Region> region)
    : variableName(std::move(variableName)), convergenceTolerance(convergenceTolerance), convergenceNorm(convergenceNorm), region(std::move(region)) {}

ablate::solver::criteria::VariableChange::~VariableChange() {
    if (previousValues) {
        VecDestroy(&previousValues) >> ablate::utilities::PetscUtilities::checkError;
    }
}

void ablate::solver::criteria::VariableChange::Initialize(const ablate::domain::Domain& domain) {
    // Get the subdomain for this region
    auto subDomain = domain.GetSubDomain(region);

    // Look up the field
    const auto& field = subDomain->GetField(variableName);

    // Get the sub vector
    IS subIs;
    DM subDm;
    Vec locVec;
    subDomain->GetFieldLocalVector(field, 0.0, &subIs, &locVec, &subDm) >> ablate::utilities::PetscUtilities::checkError;

    // Create a copy of the local vec
    VecDuplicate(locVec, &previousValues) >> ablate::utilities::PetscUtilities::checkError;
    // Set the block size as one so that it only produces one norm value
    VecSetBlockSize(previousValues, 1) >> ablate::utilities::PetscUtilities::checkError;

    // copy over the current values to use as the previous state
    VecCopy(locVec, previousValues) >> ablate::utilities::PetscUtilities::checkError;

    // restore
    subDomain->RestoreFieldLocalVector(field, &subIs, &locVec, &subDm) >> ablate::utilities::PetscUtilities::checkError;
}

bool ablate::solver::criteria::VariableChange::CheckConvergence(const ablate::domain::Domain& domain, PetscReal time, PetscInt step, const std::shared_ptr<ablate::monitors::logs::Log>& log) {
    // Get the subdomain for this region
    auto subDomain = domain.GetSubDomain(region);

    // Look up the field
    const auto& field = subDomain->GetField(variableName);

    // Get the sub vector
    IS subIs;
    DM subDm;
    Vec locVec;
    subDomain->GetFieldLocalVector(field, 0.0, &subIs, &locVec, &subDm) >> ablate::utilities::PetscUtilities::checkError;

    // Compute the norm
    PetscReal norm;
    ablate::utilities::MathUtilities::ComputeNorm(convergenceNorm, locVec, previousValues, &norm) >> ablate::utilities::PetscUtilities::checkError;

    // Create a copy of the local vec
    VecCopy(locVec, previousValues) >> ablate::utilities::PetscUtilities::checkError;

    // restore
    subDomain->RestoreFieldLocalVector(field, &subIs, &locVec, &subDm) >> ablate::utilities::PetscUtilities::checkError;

    // Check to see if converged
    bool converged = norm < convergenceTolerance;
    if (log) {
        if (converged) {
            log->Printf("\tVariableChange %s converged to %g\n", variableName.c_str(), norm);
        } else {
            log->Printf("\tVariableChange %s error: %g\n", variableName.c_str(), norm);
        }
    }

    return converged;
}

#include "registrar.hpp"
REGISTER(ablate::solver::criteria::ConvergenceCriteria, ablate::solver::criteria::VariableChange, "This class checks for a relative change in the specified variable between checks",
         ARG(std::string, "name", "the variable to check"), ARG(double, "tolerance", "the tolerance to reach to be considered converged"),
         ENUM(ablate::utilities::MathUtilities::Norm, "norm", "norm type ('l1','l1_norm','l2', 'linf', 'l2_norm')"),
         OPT(ablate::domain::Region, "region", "the region to check for a converged variable"));