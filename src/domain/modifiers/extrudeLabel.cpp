#include "extrudeLabel.hpp"
#include <petsc/private/dmimpl.h>
#include <petscdmplextransform.h>
#include "utilities/petscError.hpp"

ablate::domain::modifiers::ExtrudeLabel::ExtrudeLabel(std::vector<std::shared_ptr<domain::Region>> regions) : regions(regions) {}

std::string ablate::domain::modifiers::ExtrudeLabel::ToString() const {
    std::string string = "ablate::domain::modifiers::ExtrudeLabel: (" ;
    for(const auto& region : regions){
        string += region->ToString() + ",";
    }
    string.back() = ')';
    return string;
}

void ablate::domain::modifiers::ExtrudeLabel::Modify(DM &dm) {
    // create a temporary label to hold adapt information
    DMLabel adaptLabel;
    DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", &adaptLabel) >> checkError;

    // add points from each region
    for (const auto &region : regions) {
        region->CheckForLabel(dm);
        DMLabel regionLabel;
        PetscInt regionValue;
        domain::Region::GetLabel(region, dm, regionLabel, regionValue);

        IS bdIS;
        const PetscInt *points;
        PetscInt n, i;

        DMLabelGetStratumIS(regionLabel, regionValue, &bdIS) >> checkError;
        if (!bdIS) {
            continue;
        }
        ISGetLocalSize(bdIS, &n) >> checkError;
        ISGetIndices(bdIS, &points) >> checkError;
        for (i = 0; i < n; ++i) {
            DMLabelSetValue(adaptLabel, points[i], DM_ADAPT_REFINE) >> checkError;
        }
        ISRestoreIndices(bdIS, &points) >> checkError;
        ISDestroy(&bdIS) >> checkError;
    }

    // extrude the mesh
    DM dmAdapt;
    DMPlexTransformAdaptLabel(dm, nullptr, adaptLabel, nullptr, &dmAdapt) >> checkError;

    if (dmAdapt) {
        (dmAdapt)->prealloc_only = dm->prealloc_only; /* maybe this should go .... */
        PetscFree((dmAdapt)->vectype);
        PetscStrallocpy(dm->vectype, (char **)&(dmAdapt)->vectype);
        PetscFree((dmAdapt)->mattype);
        PetscStrallocpy(dm->mattype, (char **)&(dmAdapt)->mattype);
    }
    ReplaceDm(dm, dmAdapt);

    DMLabelDestroy(&adaptLabel) >> checkError;
}

PetscErrorCode ablate::domain::modifiers::ExtrudeLabel::DMPlexTransformAdaptLabel(DM dm, Vec metric, DMLabel adaptLabel, DMLabel rgLabel, DM *rdm) {
    DMPlexTransform tr;
    DM cdm, rcdm;
    PetscOptions petscOptions;

    PetscFunctionBegin;
    PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)dm), &tr));

    PetscOptionsCreate(&petscOptions) >> checkError;
    // build the string
    PetscOptionsInsertString(petscOptions, "-dm_plex_transform_type extrude");
    PetscCall(PetscObjectSetOptions((PetscObject)tr, petscOptions));
    PetscCall(DMPlexTransformSetDM(tr, dm));
    PetscCall(DMPlexTransformSetFromOptions(tr));
    PetscCall(DMPlexTransformSetActive(tr, adaptLabel));
    PetscCall(DMPlexTransformSetUp(tr));
    PetscCall(PetscObjectViewFromOptions((PetscObject)tr, NULL, "-dm_plex_transform_view"));
    PetscCall(DMPlexTransformApply(tr, dm, rdm));
    PetscCall(DMCopyDisc(dm, *rdm));
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetCoordinateDM(*rdm, &rcdm));
    PetscCall(DMCopyDisc(cdm, rcdm));
    PetscCall(DMPlexTransformCreateDiscLabels(tr, *rdm));
    PetscCall(DMCopyDisc(dm, *rdm));
    PetscCall(DMPlexTransformDestroy(&tr));
    PetscCall(PetscOptionsDestroy(&petscOptions));
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::ExtrudeLabel, "Extrudes a layer of cells based upon the region provided",
         ARG(std::vector<ablate::domain::Region>, "regions", "the region(s) describing the boundary cells"));