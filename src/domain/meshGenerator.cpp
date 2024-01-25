#include "meshGenerator.hpp"
#include <petsc/private/dmpleximpl.h>
#include <utility>
#include "utilities/mpiUtilities.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::domain::MeshGenerator::MeshGenerator(const std::string& name, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors,
                                             const std::shared_ptr<ablate::domain::descriptions::MeshDescription>& description, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers,
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

DM ablate::domain::MeshGenerator::CreateDM(const std::string& name, const std::shared_ptr<ablate::domain::descriptions::MeshDescription>& description) {
    // Create the new dm and set it to be a dmplex
    DM dm;
    DMCreate(PETSC_COMM_WORLD, &dm) >> utilities::MpiUtilities::checkError;
    DMSetType(dm, DMPLEX) >> utilities::MpiUtilities::checkError;

    // Get the rank
    PetscInt dim = description->GetMeshDimension();
    PetscMPIInt rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank) >> utilities::MpiUtilities::checkError;
    DMSetDimension(dm, dim) >> utilities::PetscUtilities::checkError;

    /* Must create the celltype label here so that we do not automatically try to compute the types */
    DMCreateLabel(dm, "celltype") >> utilities::PetscUtilities::checkError;

    /* Create topology */
    // Get the number of cells and numVertices from the
    auto numVertices = description->GetNumberVertices();
    auto numCells = description->GetNumberCells();

    // set the values to zero if not on the first rank
    numCells = rank == 0 ? numCells : 0;
    numVertices = rank == 0 ? numVertices : 0;
    DMPlexSetChart(dm, 0, numCells + numVertices) >> utilities::PetscUtilities::checkError;

    // Determine the max cone size and set the value for each cell.  Cells come before points
    PetscInt maxConeSize = 0;
    for (PetscInt c = 0; c < numCells; ++c) {
        auto cellType = description->GetCellType(c);

        // Determine the cone size and dim
        auto coneSize = DMPolytopeTypeGetNumVertices(cellType);
        maxConeSize = PetscMax(maxConeSize, coneSize);

        // Set the cone size
        DMPlexSetConeSize(dm, c, coneSize) >> utilities::PetscUtilities::checkError;
    }

    // With all the cones set, set up the dm
    DMSetUp(dm) >> utilities::PetscUtilities::checkError;

    // Size up a buffer for the cone
    PetscInt cone[maxConeSize];

    // Compute and set the cone for each cell
    for (PetscInt c = 0; c < numCells; ++c) {
        description->BuildTopology(c, cone);

        // Offset the cone from the number of numVertices
        for (auto& node : cone) {
            node += numCells;
        }

        auto cellType = description->GetCellType(c);
        DMPlexSetCone(dm, c, cone) >> utilities::PetscUtilities::checkError;
        DMPlexSetCellType(dm, c, cellType) >> utilities::PetscUtilities::checkError;
    }
    DMPlexSymmetrize(dm) >> utilities::PetscUtilities::checkError;
    DMPlexStratify(dm) >> utilities::PetscUtilities::checkError;

    // Now compute the location for each of the cells
    for (PetscInt v = numCells; v < numCells + numVertices; ++v) {
        DMPlexSetCellType(dm, v, DM_POLYTOPE_POINT) >> utilities::PetscUtilities::checkError;
    }
    /* Create cylinder geometry */
    {
        Vec coordinates;
        PetscSection coordSection;
        PetscScalar* coords;

        /* Build coordinates */
        DMGetCoordinateSection(dm, &coordSection) >> utilities::PetscUtilities::checkError;
        PetscSectionSetNumFields(coordSection, 1) >> utilities::PetscUtilities::checkError;
        PetscSectionSetFieldComponents(coordSection, 0, dim) >> utilities::PetscUtilities::checkError;
        PetscSectionSetChart(coordSection, numCells, numCells + numVertices) >> utilities::PetscUtilities::checkError;
        for (PetscInt v = numCells; v < numCells + numVertices; ++v) {
            PetscSectionSetDof(coordSection, v, dim) >> utilities::PetscUtilities::checkError;
            PetscSectionSetFieldDof(coordSection, v, 0, dim) >> utilities::PetscUtilities::checkError;
        }
        PetscSectionSetUp(coordSection) >> utilities::PetscUtilities::checkError;
        PetscInt coordSize;
        PetscSectionGetStorageSize(coordSection, &coordSize) >> utilities::PetscUtilities::checkError;
        VecCreate(PETSC_COMM_SELF, &coordinates) >> utilities::PetscUtilities::checkError;
        PetscObjectSetName((PetscObject)coordinates, "coordinates") >> utilities::PetscUtilities::checkError;
        VecSetSizes(coordinates, coordSize, PETSC_DETERMINE) >> utilities::PetscUtilities::checkError;
        VecSetBlockSize(coordinates, dim) >> utilities::PetscUtilities::checkError;
        VecSetType(coordinates, VECSTANDARD) >> utilities::PetscUtilities::checkError;
        VecGetArray(coordinates, &coords) >> utilities::PetscUtilities::checkError;

        // March over each vertex
        for (PetscInt v = 0; v < numVertices; ++v) {
            // Get the pointer for this coordinate
            auto vertex = coords + (v * dim);

            // Set it to zero for safety
            for (PetscInt d = 0; d < dim; ++d) {
                vertex[d] = 0.0;
            }

            // Compute the value in the description
            description->SetCoordinate(v, vertex);
        }

        VecRestoreArray(coordinates, &coords) >> utilities::PetscUtilities::checkError;
        DMSetCoordinatesLocal(dm, coordinates) >> utilities::PetscUtilities::checkError;
        VecDestroy(&coordinates) >> utilities::PetscUtilities::checkError;
    }
    /* Interpolate */
    DM idm;
    DMPlexInterpolate(dm, &idm) >> utilities::PetscUtilities::checkError;
    DMPlexCopyCoordinates(dm, idm) >> utilities::PetscUtilities::checkError;
    ReplaceDm(dm, idm);

    // set the name
    PetscObjectSetName((PetscObject)dm, name.c_str()) >> utilities::PetscUtilities::checkError;

    // label of the boundaries
    LabelBoundaries(description, dm);

    return dm;
}

void ablate::domain::MeshGenerator::LabelBoundaries(const std::shared_ptr<ablate::domain::descriptions::MeshDescription>& description, DM& dm) {
    // start by marking all the boundary faces
    auto boundaryRegion = description->GetBoundaryRegion();
    DMLabel boundaryLabel;
    PetscInt boundaryValue;
    boundaryRegion->CreateLabel(dm, boundaryLabel, boundaryValue);
    DMPlexMarkBoundaryFaces(dm, boundaryValue, boundaryLabel) >> utilities::PetscUtilities::checkError;

    // determine the new vertex/nodes bounds
    PetscInt vStart, vEnd;
    DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd) >> utilities::PetscUtilities::checkError;
    ;  // Range of vertices

    // determine the new face bounds
    PetscInt fStart, fEnd;
    DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd) >> utilities::PetscUtilities::checkError;
    ;  // Range of faces

    // now march over each face to see if it is part of another label
    std::set<PetscInt> faceSet;
    for (PetscInt f = fStart; f < fEnd; ++f) {
        // check to see if this face is in the boundary label, if not skip for now
        if (!boundaryRegion->InRegion(dm, f)) {
            continue;
        }

        // reset the faceSet
        faceSet.clear();

        // This returns everything associated with the cell in the correct ordering
        PetscInt cl, nClosure, *closure = nullptr;
        DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &nClosure, &closure) >> utilities::PetscUtilities::checkError;

        // we look at every other point in the closure because we do not need point orientations
        for (cl = 0; cl < nClosure * 2; cl += 2) {
            if (closure[cl] >= vStart && closure[cl] < vEnd) {  // Only use the points corresponding to a vertex
                faceSet.insert(closure[cl] - vStart);           // we subtract away vStart to reset the node numbering
            }
        }

        DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &nClosure, &closure) >> utilities::PetscUtilities::checkError;  // Restore the points

        // see if the face gets a label
        std::map<ablate::domain::Region, DMLabel> labels;
        if (auto region = description->GetRegion(faceSet)) {
            auto& label = labels[*region];
            PetscInt labelValue = region->GetValue();
            // create and get the label if needed
            if (!label) {
                boundaryRegion->CreateLabel(dm, label, labelValue);
            }

            // Set the value
            DMLabelSetValue(label, f, labelValue) >> utilities::PetscUtilities::checkError;
        }

        // complete each label
        for (auto& [region, label] : labels) {
            DMPlexLabelComplete(dm, label) >> utilities::PetscUtilities::checkError;
        }
    }
}
#include "registrar.hpp"
REGISTER(ablate::domain::Domain, ablate::domain::MeshGenerator, "The MeshGenerator will use a mesh description to generate an arbitrary mesh based upon supplied cells/nodes from the description.",
         ARG(std::string, "name", "the name of the domain/mesh object"), OPT(std::vector<ablate::domain::FieldDescriptor>, "fields", "a list of fields/field descriptors"),
         ARG(ablate::domain::descriptions::MeshDescription, "description", "the mesh description used to describe the cell/nodes used to create the new mesh"),
         OPT(std::vector<ablate::domain::modifiers::Modifier>, "modifiers", "a list of domain modifier"),
         OPT(ablate::parameters::Parameters, "options", "PETSc options specific to this dm.  Default value allows the dm to access global options."));
