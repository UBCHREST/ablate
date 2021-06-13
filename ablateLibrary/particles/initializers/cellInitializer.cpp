#include "cellInitializer.hpp"
#include "parser/registrar.hpp"
#include "utilities/petscError.hpp"

ablate::particles::initializers::CellInitializer::CellInitializer(int particlesPerCellPerDim) : particlesPerCell(particlesPerCellPerDim) {}

void ablate::particles::initializers::CellInitializer::Initialize(ablate::flow::Flow &flow, DM particleDm) {
    PetscInt particlesPerCell = (PetscInt)this->particlesPerCell;

    PetscInt cStart, cEnd;
    DMPlexGetHeightStratum(flow.GetDM(), 0, &cStart, &cEnd) >> checkError;
    DMSwarmSetLocalSizes(particleDm, (cEnd - cStart) * particlesPerCell, 0) >> checkError;

    // set the cell ids
    PetscInt *cellid;
    DMSwarmGetField(particleDm, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid) >> checkError;
    for (PetscInt c = cStart; c < cEnd; ++c) {
        for (PetscInt p = 0; p < particlesPerCell; ++p) {
            const PetscInt n = c * particlesPerCell + p;
            cellid[n] = c;
        }
    }

    DMSwarmRestoreField(particleDm, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid) >> checkError;
    DMSwarmSetPointCoordinatesRandom(particleDm, particlesPerCell) >> checkError;
}

REGISTER(ablate::particles::initializers::Initializer, ablate::particles::initializers::CellInitializer, "simple cell initializer that puts particles in every element",
         ARG(int, "particlesPerCellPerDim", "particles per cell per dimension"));