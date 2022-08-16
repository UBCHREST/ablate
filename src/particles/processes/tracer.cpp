#include "tracer.hpp"
#include "particles/particleSolver.hpp"
ablate::particles::processes::Tracer::Tracer(const std::string& eulerianVelocityFieldIn) : eulerianVelocityField(eulerianVelocityFieldIn.empty() ? "velocity" : eulerianVelocityFieldIn) {}

void ablate::particles::processes::Tracer::ComputeRHS(PetscReal time, ablate::particles::accessors::SwarmAccessor& swarmAccessor, ablate::particles::accessors::RhsAccessor& rhsAccessor,
                                                      ablate::particles::accessors::EulerianAccessor& eulerianAccessor) {
    auto coordinateRhs = rhsAccessor[ablate::particles::ParticleSolver::ParticleCoordinates];
    auto fluidVelocity = eulerianAccessor[eulerianVelocityField];

    // march over each particle
    const PetscInt np = swarmAccessor.GetNumberParticles();
    for (PetscInt p = 0; p < np; p++) {
        coordinateRhs.CopyFrom(fluidVelocity[p], p);
    }
}
