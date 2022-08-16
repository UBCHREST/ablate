#include "inertial.hpp"
#include "particles/particleSolver.hpp"
ablate::particles::processes::Inertial::Inertial(std::shared_ptr<parameters::Parameters> parameters, const std::string& eulerianVelocityFieldIn)
    : fluidDensity(parameters->GetExpect<PetscReal>("fluidDensity")),
      fluidViscosity(parameters->GetExpect<PetscReal>("fluidViscosity")),
      gravityField(parameters->GetExpect<std::array<PetscReal, 3>>("gravityField")),
      eulerianVelocityField(eulerianVelocityFieldIn.empty() ? "velocity" : eulerianVelocityFieldIn)

{}
void ablate::particles::processes::Inertial::ComputeRHS(PetscReal time, ablate::particles::accessors::SwarmAccessor& swarmAccessor, ablate::particles::accessors::RhsAccessor& rhsAccessor,
                                                        ablate::particles::accessors::EulerianAccessor& eulerianAccessor) {
    // Get sizes from the accessors
    const auto dim = eulerianAccessor.GetDimensions();
    const auto np = swarmAccessor.GetNumberParticles();

    PetscScalar muF = fluidViscosity;
    PetscScalar rhoF = fluidDensity;

    auto coordinateRhs = rhsAccessor[ablate::particles::ParticleSolver::ParticleCoordinates];
    auto velocityRhs = rhsAccessor[ablate::particles::ParticleSolver::ParticleVelocity];
    auto fluidVel = eulerianAccessor[eulerianVelocityField];
    auto partDiam = swarmAccessor[ablate::particles::ParticleSolver::ParticleDiameter];
    auto partVel = swarmAccessor[ablate::particles::ParticleSolver::ParticleVelocity];
    auto partDens = swarmAccessor[ablate::particles::ParticleSolver::ParticleDensity];

    for (PetscInt p = 0; p < np; ++p) {
        PetscReal rep = 0.0;
        PetscReal corFactor = 0.0;
        PetscScalar tauP;

        for (PetscInt n = 0; n < dim; n++) {
            rep += rhoF * PetscSqr(fluidVel(p, n) - partVel(p, n)) * partDiam(p) / muF;
        }
        // Correction factor to account for finite Rep on Stokes drag (see Schiller-Naumann drag closure)
        corFactor = 1.0 + 0.15 * PetscPowReal(PetscSqrtReal(rep), 0.687);
        if (rep < 0.1) {
            corFactor = 1.0;  // returns Stokes drag for low speed particles
        }
        // Note: this function assumed that the solution vector order is correct
        tauP = partDens(p) * PetscSqr(partDiam(p)) / (18.0 * muF);  // particle relaxation time
        for (PetscInt n = 0; n < dim; n++) {
            coordinateRhs(p, n) = partVel(p, n);
            velocityRhs(p, n) = corFactor * (fluidVel(p, n) - partVel(p, n)) / tauP + gravityField[n] * (1.0 - rhoF / partDens(p));
        }
    }
}
