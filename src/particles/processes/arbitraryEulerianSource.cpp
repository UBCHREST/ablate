#include "arbitraryEulerianSource.hpp"
#include "particles/particleSolver.hpp"

#include <utility>

ablate::particles::processes::ArbitraryEulerianSource::ArbitraryEulerianSource(std::string coupledFieldName, std::shared_ptr<mathFunctions::MathFunction> sourceFunction)
    : coupledFieldName(std::move(coupledFieldName)), sourceFunction(std::move(sourceFunction)) {}

void ablate::particles::processes::ArbitraryEulerianSource::ComputeEulerianSource(PetscReal startTime, PetscReal endTime, ablate::particles::accessors::SwarmAccessor& swarmAccessorPreStep,
                                                                                  ablate::particles::accessors::SwarmAccessor& swarmAccessorPostStep,
                                                                                  ablate::particles::accessors::EulerianSourceAccessor& eulerianSourceAccessor) {
    // Get sizes from the accessors
    const auto np = swarmAccessorPreStep.GetNumberParticles();
    auto coordinates = swarmAccessorPreStep[ablate::particles::ParticleSolver::ParticleCoordinates];
    auto source = eulerianSourceAccessor[coupledFieldName];
    auto dt = endTime - startTime;

    // Get the function as a petsc function
    auto function = sourceFunction->GetPetscFunction();
    auto context = sourceFunction->GetContext();

    // Store the result in a scratch variable
    PetscReal sourceValues[source.numberComponents];

    // Compute the function
    for (PetscInt p = 0; p < np; ++p) {
        function(coordinates.numberComponents, startTime, coordinates[p], source.numberComponents, sourceValues, context) >> utilities::PetscUtilities::checkError;

        // add to the source accessor and multiply by dt
        for (PetscInt c = 0; c < source.numberComponents; ++c) {
            source(p, c) += sourceValues[c] * dt;
        }
    }
}

#include "registrar.hpp"
REGISTER(ablate::particles::processes::Process, ablate::particles::processes::ArbitraryEulerianSource, "adds an arbitrary source function for each particle to the Eulerian field",
         ARG(std::string, "coupledField", "the name of the Eulerian coupled field"), ARG(ablate::mathFunctions::MathFunction, "sourceFunction", "the function to compute the source"));
