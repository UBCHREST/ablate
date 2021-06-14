#include "boxInitializer.hpp"
#include "utilities/petscError.hpp"

ablate::particles::initializers::BoxInitializer::BoxInitializer(std::vector<double> lowerBound, std::vector<double> upperBound, int particlesPerDim)
    : lowerBound(lowerBound), upperBound(upperBound), particlesPerDim(particlesPerDim){};

void ablate::particles::initializers::BoxInitializer::Initialize(ablate::flow::Flow &flow, DM particleDm) {
    /* The initial number of particles per box dimension */
    PetscInt Npb = (PetscInt)particlesPerDim;

    // determine the geom
    PetscReal partLower[3]; /* Lower left corner of particle box */
    PetscReal partUpper[3]; /* Upper right corner of particle box */

    partLower[0] = partLower[1] = partLower[2] = 0.0;
    partUpper[0] = partUpper[1] = partUpper[2] = 0.0;

    for (auto i = 0; i < PetscMin(3, lowerBound.size()); i++) {
        partLower[i] = lowerBound[i];
    }
    for (auto i = 0; i < PetscMin(3, upperBound.size()); i++) {
        partUpper[i] = upperBound[i];
    }

    PetscInt dim;
    DMGetDimension(particleDm, &dim) >> checkError;

    PetscInt Np = 1;
    PetscInt *cellid, n[3];
    PetscReal x[3], dx[3];
    PetscScalar *coords;
    PetscMPIInt rank;

    DMSetFromOptions(particleDm) >> checkError;
    MPI_Comm_rank(PetscObjectComm((PetscObject)particleDm), &rank);

    for (PetscInt d = 0; d < dim; ++d) {
        n[d] = Npb;
        dx[d] = (partUpper[d] - partLower[d]) / PetscMax(1, n[d] - 1);
        Np *= n[d];
    }

    // Initialize all particles on rank 0 and then distribute
    if (rank != 0) {
        Np = 0;
    }

    DMSwarmSetLocalSizes(particleDm, Np, 0) >> checkError;
    DMSetFromOptions(particleDm) >> checkError;
    DMSwarmGetField(particleDm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords) >> checkError;
    if (rank == 0) {
        switch (dim) {
            case 2:
                x[0] = partLower[0];
                for (PetscInt i = 0; i < n[0]; ++i, x[0] += dx[0]) {
                    x[1] = partLower[1];
                    for (PetscInt j = 0; j < n[1]; ++j, x[1] += dx[1]) {
                        const PetscInt p = j * n[0] + i;
                        for (PetscInt d = 0; d < dim; ++d) {
                            coords[p * dim + d] = x[d];
                        }
                    }
                }
                break;
            case 3:
                x[0] = partLower[0];
                for (PetscInt i = 0; i < n[0]; ++i, x[0] += dx[0]) {
                    x[1] = partLower[1];
                    for (PetscInt j = 0; j < n[1]; ++j, x[1] += dx[1]) {
                        x[2] = partLower[2];
                        for (PetscInt k = 0; k < n[2]; ++k, x[2] += dx[2]) {
                            const PetscInt p = (k * n[1] + j) * n[0] + i;
                            for (PetscInt d = 0; d < dim; ++d) {
                                coords[p * dim + d] = x[d];
                            }
                        }
                    }
                }
                break;
            default:
                throw std::runtime_error("Do not support particle layout in dimension " + std::to_string(dim));
        }
    }
    DMSwarmRestoreField(particleDm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords) >> checkError;
    DMSwarmGetField(particleDm, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid) >> checkError;
    for (PetscInt p = 0; p < Np; ++p) {
        cellid[p] = 0;
    }
    DMSwarmRestoreField(particleDm, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid) >> checkError;
    DMSwarmMigrate(particleDm, PETSC_TRUE) >> checkError;
}

#include "parser/registrar.hpp"

REGISTER(ablate::particles::initializers::Initializer, ablate::particles::initializers::BoxInitializer, "simple box initializer that puts particles in a defined box",
         ARG(std::vector<double>, "lower", "the lower bound of the box"), ARG(std::vector<double>, "upper", "the upper bound of the box"),
         ARG(int, "particlesPerDim", "the particles per box dimension"));