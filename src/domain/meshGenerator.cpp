#include "meshGenerator.hpp"
#include <petsc/private/dmpleximpl.h>
#include "utilities/mpiUtilities.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::domain::MeshGenerator::MeshGenerator(const std::string& name, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors,
                                             std::shared_ptr<ablate::domain::descriptions::MeshDescription> description, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers,
                                             const std::shared_ptr<parameters::Parameters>& options)
    : Domain(CreateDM(name, description), name, std::move(fieldDescriptors), std::move(modifiers), options) {}

ablate::domain::MeshGenerator::~MeshGenerator() {
    if (dm) {
        DMDestroy(&dm);
    }
}

void ablate::domain::MeshGenerator::ReplaceDm(DM& originalDm, DM& replaceDm) {
    if (replaceDm) {
        // copy over the name
        const char* name;
        PetscObjectName((PetscObject)originalDm) >> utilities::PetscUtilities::checkError;
        PetscObjectGetName((PetscObject)originalDm, &name) >> utilities::PetscUtilities::checkError;
        PetscObjectSetName((PetscObject)replaceDm, name) >> utilities::PetscUtilities::checkError;

        // Copy over the options object
        PetscOptions options;
        PetscObjectGetOptions((PetscObject)originalDm, &options) >> utilities::PetscUtilities::checkError;
        PetscObjectSetOptions((PetscObject)replaceDm, options) >> utilities::PetscUtilities::checkError;
        ((DM_Plex*)(replaceDm)->data)->useHashLocation = ((DM_Plex*)originalDm->data)->useHashLocation;

        DMDestroy(&originalDm) >> utilities::PetscUtilities::checkError;
        originalDm = replaceDm;
    }
}

DM ablate::domain::MeshGenerator::CreateDM(const std::string& name, std::shared_ptr<ablate::domain::descriptions::MeshDescription> description) {
    // Create the new dm and set it to be a dmplex
    DM dm;
    const PetscInt dim = 3;
    DMCreate(PETSC_COMM_WORLD, &dm) >> utilities::MpiUtilities::checkError;
    DMSetType(dm, DMPLEX) >> utilities::MpiUtilities::checkError;

    // Get the rank
    PetscMPIInt rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank) >> utilities::MpiUtilities::checkError;
    DMSetDimension(dm, dim) >> utilities::PetscUtilities::checkError;

    /* Must create the celltype label here so that we do not automatically try to compute the types */
    DMCreateLabel(dm, "celltype") >> utilities::PetscUtilities::checkError;
    /* Create topology */
    //    {
    //        // Get the number of cells and numVertices from the
    //        auto numVertices = description->GetNumberVertices();
    //        auto numCells = description->GetNumberCells();
    //
    //            PetscInt cone[6],c;
    //
    //        numCells = rank == 0 ? numberWedges : 0;
    //        numVertices = rank == 0 ? 2 * (numberWedges + 1) : 0;
    //        DMPlexSetChart(dm, 0, numCells + numVertices) >> utilities::PetscUtilities::checkError;
    //        for (c = 0; c < numCells; c++) DMPlexSetConeSize(dm, c, 6) >> utilities::PetscUtilities::checkError;
    //        DMSetUp(dm) >> utilities::PetscUtilities::checkError;
    //        for (c = 0; c < numCells; c++) {
    //            cone[0] = c + numberWedges * 1;
    //            cone[1] = (c + 1) % numberWedges + numberWedges * 1;
    //            cone[2] = 0 + 3 * numberWedges;
    //            cone[3] = c + numberWedges * 2;
    //            cone[4] = (c + 1) % numberWedges + numberWedges * 2;
    //            cone[5] = 1 + 3 * numberWedges;
    //            std::cout << "Cone: " << ablate::utilities::VectorUtilities::Concatenate(cone, 6) << std::endl;
    //            DMPlexSetCone(dm, c, cone) >> utilities::PetscUtilities::checkError;
    //            DMPlexSetCellType(dm, c, DM_POLYTOPE_TRI_PRISM) >> utilities::PetscUtilities::checkError;
    //        }
    //        DMPlexSymmetrize(dm) >> utilities::PetscUtilities::checkError;
    //        DMPlexStratify(dm) >> utilities::PetscUtilities::checkError;
    //    }
    //    for (v = numCells; v < numCells + numVertices; ++v) DMPlexSetCellType(dm, v, DM_POLYTOPE_POINT) >> utilities::PetscUtilities::checkError;
    //    /* Create cylinder geometry */
    //    {
    //        Vec coordinates;
    //        PetscSection coordSection;
    //        PetscScalar* coords;
    //        PetscInt coordSize, c;
    //
    //        /* Build coordinates */
    //        DMGetCoordinateSection(dm, &coordSection) >> utilities::PetscUtilities::checkError;
    //        PetscSectionSetNumFields(coordSection, 1) >> utilities::PetscUtilities::checkError;
    //        PetscSectionSetFieldComponents(coordSection, 0, dim) >> utilities::PetscUtilities::checkError;
    //        PetscSectionSetChart(coordSection, numCells, numCells + numVertices) >> utilities::PetscUtilities::checkError;
    //        for (v = numCells; v < numCells + numVertices; ++v) {
    //            PetscSectionSetDof(coordSection, v, dim) >> utilities::PetscUtilities::checkError;
    //            PetscSectionSetFieldDof(coordSection, v, 0, dim) >> utilities::PetscUtilities::checkError;
    //        }
    //        PetscSectionSetUp(coordSection) >> utilities::PetscUtilities::checkError;
    //        PetscSectionGetStorageSize(coordSection, &coordSize) >> utilities::PetscUtilities::checkError;
    //        VecCreate(PETSC_COMM_SELF, &coordinates) >> utilities::PetscUtilities::checkError;
    //        PetscObjectSetName((PetscObject)coordinates, "coordinates") >> utilities::PetscUtilities::checkError;
    //        VecSetSizes(coordinates, coordSize, PETSC_DETERMINE) >> utilities::PetscUtilities::checkError;
    //        VecSetBlockSize(coordinates, dim) >> utilities::PetscUtilities::checkError;
    //        VecSetType(coordinates, VECSTANDARD) >> utilities::PetscUtilities::checkError;
    //        VecGetArray(coordinates, &coords) >> utilities::PetscUtilities::checkError;
    //        for (c = 0; c < numCells; c++) {
    //            coords[(c + 0 * numberWedges) * dim + 0] = PetscCosReal(2.0 * c * PETSC_PI / numberWedges);
    //            coords[(c + 0 * numberWedges) * dim + 1] = PetscSinReal(2.0 * c * PETSC_PI / numberWedges);
    //            coords[(c + 0 * numberWedges) * dim + 2] = 1.0;
    //            coords[(c + 1 * numberWedges) * dim + 0] = PetscCosReal(2.0 * c * PETSC_PI / numberWedges);
    //            coords[(c + 1 * numberWedges) * dim + 1] = PetscSinReal(2.0 * c * PETSC_PI / numberWedges);
    //            coords[(c + 1 * numberWedges) * dim + 2] = 0.0;
    //            std::cout << "Vertex: " << ((c + 0 * numberWedges) * dim) << "," << coords[(c + 0 * numberWedges) * dim + 0] << ", " << coords[(c + 0 * numberWedges) * dim + 1] << ", "
    //                      << coords[(c + 0 * numberWedges) * dim + 2] << std::endl;
    //            std::cout << "Vertex: " << ((c + 1 * numberWedges) * dim) << "," << coords[(c + 1 * numberWedges) * dim + 0] << ", " << coords[(c + 1 * numberWedges) * dim + 1] << ", "
    //                      << coords[(c + 1 * numberWedges) * dim + 2] << std::endl;
    //        }
    //        if (rank == 0) {
    //            coords[(2 * numberWedges + 0) * dim + 0] = 0.0;
    //            coords[(2 * numberWedges + 0) * dim + 1] = 0.0;
    //            coords[(2 * numberWedges + 0) * dim + 2] = 1.0;
    //            coords[(2 * numberWedges + 1) * dim + 0] = 0.0;
    //            coords[(2 * numberWedges + 1) * dim + 1] = 0.0;
    //            coords[(2 * numberWedges + 1) * dim + 2] = 0.0;
    //            std::cout << "Vertex: " << ((2 * numberWedges + 0) * dim) << "," << coords[(2 * numberWedges + 0) * dim + 0] << ", " << coords[(2 * numberWedges + 0) * dim + 1] << ", "
    //                      << coords[(2 * numberWedges + 0) * dim + 2] << std::endl;
    //            std::cout << "Vertex: " << ((2 * numberWedges + 1) * dim) << "," << coords[(2 * numberWedges + 1) * dim + 0] << ", " << coords[(2 * numberWedges + 1) * dim + 1] << ", "
    //                      << coords[(2 * numberWedges + 1) * dim + 2] << std::endl;
    //        }
    //        VecRestoreArray(coordinates, &coords) >> utilities::PetscUtilities::checkError;
    //        DMSetCoordinatesLocal(dm, coordinates) >> utilities::PetscUtilities::checkError;
    //        VecDestroy(&coordinates) >> utilities::PetscUtilities::checkError;
    //    }
    /* Interpolate */
    DM idm;
    DMPlexInterpolate(dm, &idm) >> utilities::PetscUtilities::checkError;
    DMPlexCopyCoordinates(dm, idm) >> utilities::PetscUtilities::checkError;
    ReplaceDm(dm, idm);

    return dm;
}
