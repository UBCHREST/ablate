#include "boxInitializer.hpp"

#include <utility>
#include "utilities/petscUtilities.hpp"

ablate::particles::initializers::BoxInitializer::BoxInitializer(std::vector<double> lowerBound, std::vector<double> upperBound, int particlesPerDim)
    : lowerBound(std::move(lowerBound)), upperBound(std::move(upperBound)), particlesPerDim(particlesPerDim) {}

void ablate::particles::initializers::BoxInitializer::Initialize(ablate::domain::SubDomain &flow, DM particleDm) {
    /* The initial number of particles per box dimension */
    auto Npb = (PetscInt)particlesPerDim;

    // determine the geom
    PetscReal partLower[3]; /* Lower left corner of particle box */
    PetscReal partUpper[3]; /* Upper right corner of particle box */

    partLower[0] = partLower[1] = partLower[2] = 0.0;
    partUpper[0] = partUpper[1] = partUpper[2] = 0.0;

    for (std::size_t i = 0; i < PetscMin(3, lowerBound.size()); i++) {
        partLower[i] = lowerBound[i];
    }
    for (std::size_t i = 0; i < PetscMin(3, upperBound.size()); i++) {
        partUpper[i] = upperBound[i];
    }

    PetscInt dim;
    DMGetDimension(particleDm, &dim) >> utilities::PetscUtilities::checkError;

    PetscInt Np = 1;
    PetscInt Nfc;
    DMSwarmCellDM cellDm;
    PetscInt *swarm_cellid, n[3];
    PetscReal x[3], dx[3];
    PetscScalar *coords;
    const char **coordFields, *cellid;
    PetscMPIInt rank;

    DMSetFromOptions(particleDm) >> utilities::PetscUtilities::checkError;
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

    DMSwarmSetLocalSizes(particleDm, Np, 0) >> utilities::PetscUtilities::checkError;
    DMSetFromOptions(particleDm) >> utilities::PetscUtilities::checkError;

    DMSwarmGetCellDMActive(particleDm, &cellDm) >> utilities::PetscUtilities::checkError;
    DMSwarmCellDMGetCellID(cellDm, &cellid) >> utilities::PetscUtilities::checkError;
    DMSwarmCellDMGetCoordinateFields(cellDm, &Nfc, &coordFields);
    if (!(Nfc == 1)) throw std::runtime_error("NFc shouldn't be 1 i don't think");
    DMSwarmGetField(particleDm, coordFields[0], NULL, NULL, (void **)&coords) >> utilities::PetscUtilities::checkError;
//    DMSwarmGetField(particleDm, DMSwarmPICField_coor, nullptr, nullptr, (void **)&coords) >> utilities::PetscUtilities::checkError;

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
    DMSwarmRestoreField(particleDm, coordFields[0], NULL, NULL, (void **)&coords) >> utilities::PetscUtilities::checkError;
//    DMSwarmRestoreField(particleDm, DMSwarmPICField_coor, nullptr, nullptr, (void **)&coords) >> utilities::PetscUtilities::checkError;

    DMSwarmGetField(particleDm, cellid, NULL, NULL, (void **)&swarm_cellid) >> utilities::PetscUtilities::checkError;
//    DMSwarmGetField(particleDm, DMSwarmPICField_cellid, nullptr, nullptr, (void **)&cellid) >> utilities::PetscUtilities::checkError;
    for (PetscInt p = 0; p < Np; ++p) {
        swarm_cellid[p] = 0;
    }
    DMSwarmRestoreField(particleDm, cellid, NULL, NULL, (void **)&swarm_cellid) >> utilities::PetscUtilities::checkError;
//    DMSwarmRestoreField(particleDm, DMSwarmPICField_cellid, nullptr, nullptr, (void **)&cellid) >> utilities::PetscUtilities::checkError;
    DMSwarmMigrate(particleDm, PETSC_TRUE) >> utilities::PetscUtilities::checkError;
}

#include "registrar.hpp"
REGISTER(ablate::particles::initializers::Initializer, ablate::particles::initializers::BoxInitializer, "simple box initializer that puts particles in a defined box",
         ARG(std::vector<double>, "lower", "the lower bound of the box"), ARG(std::vector<double>, "upper", "the upper bound of the box"),
         ARG(int, "particlesPerDim", "the particles per box dimension"));