static const char help[] = "Extrudes the nozzle example";

#include <petscdmplex.h>
#include <petsc.h>
#include <petscoptions.h>
static PetscErrorCode xyz(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx) {
    for (PetscInt i = 0; i < Nf; i++) {
        u[i] = x[i];
    }
    return 0;
}

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    // download the nozzle file
    PetscBool found;
    char localPath[PETSC_MAX_PATH_LEN];
    char fileUrl[PETSC_MAX_PATH_LEN];
    PetscCall(PetscOptionsGetString(NULL, NULL, "-filename", fileUrl, PETSC_MAX_PATH_LEN, NULL));
    PetscCall(PetscFileRetrieve(PETSC_COMM_WORLD, fileUrl, localPath, PETSC_MAX_PATH_LEN, &found));
    if (!found) {
        SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, "Unable to locate mesh file");
    }

    // Create the surface
    DM surface;
    PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, localPath, "surfaceMesh", PETSC_TRUE, &surface));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)surface, "sur_"));
    PetscCall(DMSetFromOptions(surface));
    PetscCall(DMViewFromOptions(surface, NULL, "-dm_view"));

    // Create the volume
    DM dm;
    PetscCall(DMPlexGenerate(surface, "tetgen", PETSC_TRUE, &dm));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)surface, "org_"));
    PetscCall(PetscObjectSetName((PetscObject)dm, "orgMesh"));
    PetscCall(DMPlexSetRefinementUniform(dm, PETSC_TRUE));

    PetscCall(DMPlexInflateToGeomModel(dm));

    PetscCall(DMSetFromOptions(dm));
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

    // mark the boundary faces
    DMLabel marker;
    PetscCall(DMCreateLabel(dm, "marker"));
    PetscCall(DMGetLabel(dm, "marker", &marker));
    PetscCall(DMPlexMarkBoundaryFaces(dm, 1, marker));
    PetscCall(DMPlexLabelComplete(dm, marker));

    // now extrude boundary
    DM dma;
    PetscCall(DMAdaptLabel(dm, marker, &dma));
    PetscCall(PetscObjectSetName((PetscObject)dma, "adaptedMeshMesh"));
    PetscCall(DMViewFromOptions(dma, NULL, "-adapt_dm_view"));

    // clean up
    PetscCall(DMDestroy(&dm));
    PetscCall(DMDestroy(&surface));
    PetscCall(DMDestroy(&dma));
    PetscCall(PetscFinalize());
    return 0;
}

/*TEST

    args: -dm_adaptor cellrefiner -dm_plex_transform_type extrude -sur_dm_refine 2 -dm_view -sur_dm_view -adapt_dm_view \
    -dm_plex_transform_extrude_thickness 0.0001 -dm_plex_transform_extrude_use_tensor 0 \
    -filename /Users/mcgurn/scratch/ablate/tests/integrationTests/inputs/mesh/cylinder.stp


TEST*/
