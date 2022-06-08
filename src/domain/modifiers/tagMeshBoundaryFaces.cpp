#include "tagMeshBoundaryFaces.hpp"
#include <utilities/petscError.hpp>
#include <utility>

ablate::domain::modifiers::TagMeshBoundaryFaces::TagMeshBoundaryFaces(std::shared_ptr<domain::Region> region) : region(std::move(region)) {}

void ablate::domain::modifiers::TagMeshBoundaryFaces::Modify(DM &dm) {
    // Create a new label
    DMCreateLabel(dm, region->GetName().c_str()) >> checkError;
    DMLabel boundaryFaceLabel;
    DMGetLabel(dm, region->GetName().c_str(), &boundaryFaceLabel) >> checkError;

    // mark the boundary faces
    DMPlexMarkBoundaryFaces(dm, region->GetValue(), boundaryFaceLabel) >> checkError;
    DMPlexLabelComplete(dm, boundaryFaceLabel) >> checkError;
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::TagMeshBoundaryFaces, "Mark/tag all faces on the boundary of the mesh",
         ARG(ablate::domain::Region, "region", "the new region for the boundary faces"));