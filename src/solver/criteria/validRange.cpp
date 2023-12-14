#include "validRange.hpp"
#include "convergenceException.hpp"

#include <domain/subDomain.hpp>
#include <utility>

ablate::solver::criteria::ValidRange::ValidRange(std::string variableName, double lowerBound, double upperBound, std::shared_ptr<ablate::domain::Region> region)
    : variableName(std::move(variableName)), lowerBound(lowerBound), upperBound(upperBound), region(std::move(region)) {}

bool ablate::solver::criteria::ValidRange::CheckConvergence(const ablate::domain::Domain& domain, PetscReal time, PetscInt step, const std::shared_ptr<ablate::monitors::logs::Log>& log) {
    // Get the subdomain for this region
    auto subDomain = domain.GetSubDomain(region);

    // Look up the field
    const auto& field = subDomain->GetField(variableName);

    // Get the sub vector
    IS subIs;
    DM subDm;
    Vec locVec;
    subDomain->GetFieldLocalVector(field, 0.0, &subIs, &locVec, &subDm) >> ablate::utilities::PetscUtilities::checkError;

    // Get the min and max values
    PetscScalar min, max;
    VecMin(locVec, nullptr, &min) >> ablate::utilities::PetscUtilities::checkError;
    VecMax(locVec, nullptr, &max) >> ablate::utilities::PetscUtilities::checkError;

    // restore
    subDomain->RestoreFieldLocalVector(field, &subIs, &locVec, &subDm) >> ablate::utilities::PetscUtilities::checkError;

    // check upper and lower bounds
    if (max < lowerBound || min > upperBound) {
        std::string message = max < lowerBound ? variableName + " all values fall below the the lower bound " + std::to_string(lowerBound) + ". Maximum value is " + std::to_string(max)
                                               : variableName + " all values exceed the the upper bound " + std::to_string(upperBound) + ". Minimum value is " + std::to_string(min);
        if (log) {
            log->Printf("%s\n", message.c_str());
        }
        throw ConvergenceException(message);
    }

    return true;
}

#include "registrar.hpp"
REGISTER(ablate::solver::criteria::ConvergenceCriteria, ablate::solver::criteria::ValidRange, "This class will stop the convergence iterations if the bounds are exceeded",
         ARG(std::string, "name", "the variable to check"), ARG(double, "lowerBound", "values lower than this will result in an exception"),
         ARG(double, "upperBound", "values higher than this will result in an exception"), OPT(ablate::domain::Region, "region", "the region to check for a converged variable"));