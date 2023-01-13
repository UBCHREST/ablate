#include "extrudeLabel.hpp"
#include <petsc/private/dmimpl.h>
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

    // Determine the current max cell int
    PetscInt originalMaxCell;
    DMPlexGetHeightStratum(dm, 0, nullptr, &originalMaxCell) >> utilities::PetscUtilities::checkError;

    // March over each cell in this rank and determine if it is original or not
    PetscInt cStart, cEnd;
    DMPlexGetHeightStratum(dmAdapt, 0, &cStart, &cEnd) >> utilities::PetscUtilities::checkError;
    for (PetscInt c = cStart; c < cEnd; ++c) {
        if (c < originalMaxCell) {
            DMLabelSetValue(originalRegionLabel, c, originalRegionValue) >> utilities::PetscUtilities::checkError;
        } else {
            DMLabelSetValue(extrudedRegionLabel, c, extrudedRegionValue) >> utilities::PetscUtilities::checkError;
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
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::ExtrudeLabel, "Extrudes a layer of cells based upon the region provided",
         ARG(std::vector<ablate::domain::Region>, "regions", "the region(s) describing the boundary cells"),
         ARG(ablate::domain::Region, "boundaryRegion", "the new label describing the faces between the original and extruded regions"),
         ARG(ablate::domain::Region, "originalRegion", "the region describing the original mesh"), ARG(ablate::domain::Region, "extrudedRegion", "the region describing the new extruded cells"),
         OPT(double, "thickness", "thickness for the extruded cells. If default (0) the 2 * minimum cell radius is used"));