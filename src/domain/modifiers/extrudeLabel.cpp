#include "extrudeLabel.hpp"
#include <petsc/private/dmimpl.h>
#include <petsc/private/petscfeimpl.h>
#include <petscdmplextransform.h>
#include <utility>
#include "tagLabelInterface.hpp"
#include "utilities/petscUtilities.hpp"

ablate::domain::modifiers::ExtrudeLabel::ExtrudeLabel(std::vector<std::shared_ptr<domain::Region>> regions, std::shared_ptr<domain::Region> boundaryRegion,
                                                      std::shared_ptr<domain::Region> originalRegion, std::shared_ptr<domain::Region> extrudedRegion, double thickness)
    : regions(std::move(regions)), boundaryRegion(std::move(std::move(boundaryRegion))), originalRegion(std::move(originalRegion)), extrudedRegion(std::move(extrudedRegion)), thickness(thickness) {}

std::string ablate::domain::modifiers::ExtrudeLabel::ToString() const {
    std::string string = "ablate::domain::modifiers::ExtrudeLabel: (";
    for (const auto &region : regions) {
        string += region->ToString() + ",";
    }
    string.back() = ')';
    return string;
}

void ablate::domain::modifiers::ExtrudeLabel::Modify(DM &dm) {
    // create a temporary label to hold adapt information
    DMLabel adaptLabel;
    DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", &adaptLabel) >> utilities::PetscUtilities::checkError;

    // add points from each region
    for (const auto &region : regions) {
        region->CheckForLabel(dm, PetscObjectComm((PetscObject)dm));
        DMLabel regionLabel;
        PetscInt regionValue;
        domain::Region::GetLabel(region, dm, regionLabel, regionValue);

        // If this label exists on this domain
        if (regionLabel) {
            IS bdIS;
            const PetscInt *points;
            PetscInt n, i;

            DMLabelGetStratumIS(regionLabel, regionValue, &bdIS) >> utilities::PetscUtilities::checkError;
            if (!bdIS) {
                continue;
            }
            ISGetLocalSize(bdIS, &n) >> utilities::PetscUtilities::checkError;
            ISGetIndices(bdIS, &points) >> utilities::PetscUtilities::checkError;
            for (i = 0; i < n; ++i) {
                DMLabelSetValue(adaptLabel, points[i], DM_ADAPT_REFINE) >> utilities::PetscUtilities::checkError;
            }
            ISRestoreIndices(bdIS, &points) >> utilities::PetscUtilities::checkError;
            ISDestroy(&bdIS) >> utilities::PetscUtilities::checkError;
        }
    }

    // set the options for the transform
    PetscOptions transformOptions;
    PetscOptionsCreate(&transformOptions) >> utilities::PetscUtilities::checkError;
    PetscOptionsInsertString(transformOptions, "-dm_plex_transform_type extrude");
    PetscOptionsInsertString(transformOptions, "-dm_plex_transform_extrude_use_tensor 0");

    // determine if the thickness needs to be computed
    PetscReal extrudeThickness = thickness;
    if (extrudeThickness == 0.0) {
        // Get the fv geom
        DMPlexGetGeometryFVM(dm, nullptr, nullptr, &extrudeThickness) >> utilities::PetscUtilities::checkError;
        extrudeThickness *= 2.0;  // double the thickness
    }
    const auto extrudeThicknessString = std::to_string(extrudeThickness);
    PetscOptionsSetValue(transformOptions, "-dm_plex_transform_extrude_thickness", extrudeThicknessString.c_str());

    // extrude the mesh
    DM dmAdapt;
    DMPlexTransformAdaptLabel(dm, nullptr, adaptLabel, nullptr, transformOptions, &dmAdapt) >> utilities::PetscUtilities::checkError;

    if (dmAdapt) {
        (dmAdapt)->prealloc_only = dm->prealloc_only; /* maybe this should go .... */
        PetscFree((dmAdapt)->vectype);
        PetscStrallocpy(dm->vectype, (char **)&(dmAdapt)->vectype);
        PetscFree((dmAdapt)->mattype);
        PetscStrallocpy(dm->mattype, (char **)&(dmAdapt)->mattype);
    }

    // create hew new labels for each region (on the new adapted dm)
    DMLabel originalRegionLabel, extrudedRegionLabel;
    PetscInt originalRegionValue, extrudedRegionValue;
    originalRegion->CreateLabel(dmAdapt, originalRegionLabel, originalRegionValue);
    extrudedRegion->CreateLabel(dmAdapt, extrudedRegionLabel, extrudedRegionValue);

    // March over each cell in this rank and determine if it is original or not
    PetscInt cAdaptStart, cAdaptEnd;
    DMPlexGetHeightStratum(dmAdapt, 0, &cAdaptStart, &cAdaptEnd) >> utilities::PetscUtilities::checkError;

    // cell the depths of the cell layer
    PetscInt cellDepth;
    DMPlexGetDepth(dm, &cellDepth) >> utilities::PetscUtilities::checkError;
    DMLabel depthAdaptLabel, ctAdaptLabel, ctLabel;
    DMPlexGetDepthLabel(dmAdapt, &depthAdaptLabel) >> utilities::PetscUtilities::checkError;
    DMPlexGetCellTypeLabel(dmAdapt, &ctAdaptLabel) >> utilities::PetscUtilities::checkError;
    DMPlexGetCellTypeLabel(dm, &ctLabel) >> utilities::PetscUtilities::checkError;

    // because the new cells can be intertwined with the old cells for mixed use we need to do this cell type by cell type
    for (PetscInt cellType = 0; cellType < DM_NUM_POLYTOPES; ++cellType) {
        auto ict = (DMPolytopeType)cellType;

        // get the new range for this cell type
        PetscInt tAdaptStart, tAdaptEnd;
        DMLabelGetStratumBounds(ctAdaptLabel, ict, &tAdaptStart, &tAdaptEnd) >> utilities::PetscUtilities::checkError;

        // only check if there are cell of this type
        if (tAdaptStart < 0) {
            continue;
        }
        // determine the depth of this cell type
        PetscInt cellTypeDepth;
        DMLabelGetValue(depthAdaptLabel, tAdaptStart, &cellTypeDepth) >> utilities::PetscUtilities::checkError;
        if (cellTypeDepth != cellDepth) {
            continue;
        }

        // Get the original range for this cell type
        PetscInt tStart, tEnd;
        DMLabelGetStratumBounds(ctLabel, ict, &tStart, &tEnd) >> utilities::PetscUtilities::checkError;
        PetscInt numberOldCells = tStart >= 0 ? tEnd - tStart : 0;

        // march over each new cell
        for (PetscInt c = tAdaptStart; c < tAdaptEnd; ++c) {
            if ((c - tAdaptStart) < numberOldCells) {
                DMLabelSetValue(originalRegionLabel, c, originalRegionValue) >> utilities::PetscUtilities::checkError;
            } else {
                DMLabelSetValue(extrudedRegionLabel, c, extrudedRegionValue) >> utilities::PetscUtilities::checkError;
            }
        }
    }

    // complete the labels
    DMPlexLabelComplete(dmAdapt, originalRegionLabel);
    DMPlexLabelComplete(dmAdapt, extrudedRegionLabel);

    // tag the interface between the faces (reuse modifier)
    TagLabelInterface(originalRegion, extrudedRegion, boundaryRegion).Modify(dmAdapt);

    // replace the dm
    ReplaceDm(dm, dmAdapt);

    // cleanup
    PetscOptionsDestroy(&transformOptions) >> utilities::PetscUtilities::checkError;
    DMLabelDestroy(&adaptLabel) >> utilities::PetscUtilities::checkError;
}

PetscErrorCode ablate::domain::modifiers::ExtrudeLabel::DMPlexTransformAdaptLabel(DM dm, Vec, DMLabel adaptLabel, DMLabel, PetscOptions transformOptions, DM *rdm) {
    DMPlexTransform tr;
    DM cdm, rcdm;

    PetscFunctionBegin;
    PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)dm), &tr));
    PetscCall(PetscObjectSetOptions((PetscObject)tr, transformOptions));
    PetscCall(DMPlexTransformSetDM(tr, dm));
    PetscCall(DMPlexTransformSetFromOptions(tr));
    PetscCall(DMPlexTransformSetActive(tr, adaptLabel));
    PetscCall(DMPlexTransformSetUp(tr));
    PetscCall(PetscObjectViewFromOptions((PetscObject)tr, nullptr, "-dm_plex_transform_view"));
    PetscCall(DMPlexTransformApply(tr, dm, rdm));
    PetscCall(DMCopyDisc(dm, *rdm));
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetCoordinateDM(*rdm, &rcdm));
    PetscCall(DMCopyDisc(cdm, rcdm));
    PetscCall(DMPlexTransformCreateDiscLabels(tr, *rdm));
    PetscCall(DMCopyDisc(dm, *rdm));
    PetscCall(DMPlexTransformDestroy(&tr));
    ((DM_Plex *)(*rdm)->data)->useHashLocation = ((DM_Plex *)dm->data)->useHashLocation;
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::ExtrudeLabel, "Extrudes a layer of cells based upon the region provided",
         ARG(std::vector<ablate::domain::Region>, "regions", "the region(s) describing the boundary cells"),
         ARG(ablate::domain::Region, "boundaryRegion", "the new label describing the faces between the original and extruded regions"),
         ARG(ablate::domain::Region, "originalRegion", "the region describing the original mesh"), ARG(ablate::domain::Region, "extrudedRegion", "the region describing the new extruded cells"),
         OPT(double, "thickness", "thickness for the extruded cells. If default (0) the 2 * minimum cell radius is used"));