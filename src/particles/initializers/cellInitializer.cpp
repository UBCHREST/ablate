#include "cellInitializer.hpp"
#include "utilities/petscUtilities.hpp"

ablate::particles::initializers::CellInitializer::CellInitializer(int particlesPerCellPerDim) : particlesPerCell(particlesPerCellPerDim) {}

void ablate::particles::initializers::CellInitializer::Initialize(ablate::domain::SubDomain &flow, DM particleDm) {
    auto particlesPerCellLocal = (PetscInt)this->particlesPerCell;

    PetscInt cStart, cEnd;
    DMPlexGetHeightStratum(flow.GetDM(), 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;
    DMSwarmSetLocalSizes(particleDm, (cEnd - cStart) * particlesPerCellLocal, 0) >> utilities::PetscUtilities::checkError;

    // set the cell ids
    PetscInt *cellid;
    DMSwarmGetField(particleDm, DMSwarmPICField_cellid, nullptr, nullptr, (void **)&cellid) >> utilities::PetscUtilities::checkError;
    for (PetscInt c = cStart; c < cEnd; ++c) {
        for (PetscInt p = 0; p < particlesPerCellLocal; ++p) {
            const PetscInt n = c * particlesPerCellLocal + p;
            cellid[n] = c;
        }
    }

    DMSwarmRestoreField(particleDm, DMSwarmPICField_cellid, nullptr, nullptr, (void **)&cellid) >> utilities::PetscUtilities::checkError;
    DMSwarmSetPointCoordinatesRandom(particleDm, particlesPerCellLocal) >> utilities::PetscUtilities::checkError;
}

#include "registrar.hpp"
REGISTER(ablate::particles::initializers::Initializer, ablate::particles::initializers::CellInitializer, "simple cell initializer that puts particles in every element",
         ARG(int, "particlesPerCellPerDim", "particles per cell per dimension"));