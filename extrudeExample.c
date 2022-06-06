static const char help[] = "Extrudes the nozzle example";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));


    // download the nozzle file
    PetscBool found;
    char localPath[PETSC_MAX_PATH_LEN];
    char fileUrl[] = "https://gitlab.com/petsc/petsc/-/raw/main/share/petsc/datafiles/meshes/nozzle.igs";
    PetscCall(PetscFileRetrieve(PETSC_COMM_WORLD, fileUrl, localPath, PETSC_MAX_PATH_LEN, &found));
    if(!found){
        SETERRABORT(PETSC_COMM_WORLD, PETSC_ERR_FILE_OPEN, "Unable to locate mesh file" );
    }

    // Create the surface
    DM surface;
    PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, localPath, "exampleExtrudeSurface", PETSC_TRUE, &surface));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) surface, "sur_"));
    PetscCall(DMSetFromOptions(surface));
    PetscCall(DMViewFromOptions(surface, NULL, "-dm_view"));

    // Create the volume
    DM dm;
    PetscCall(DMPlexGenerate(surface, "tetgen", PETSC_TRUE, &dm));
    PetscCall(PetscObjectSetName((PetscObject) dm, "CAD Mesh"));
    PetscCall(DMPlexSetRefinementUniform(dm, PETSC_TRUE));
    PetscCall(DMViewFromOptions(dm, NULL, "-pre_dm_view"));

    PetscCall(DMPlexInflateToGeomModel(dm));
    PetscCall(DMViewFromOptions(dm, NULL, "-inf_dm_view"));

    PetscCall(DMSetFromOptions(dm));
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

    // mark the boundary faces
    DMLabel marker;
    PetscCall(DMCreateLabel(dm, "marker"));
    PetscCall(DMGetLabel(dm, "marker", &marker));
    PetscCall(DMPlexMarkBoundaryFaces(dm, 1, marker));

    // now extrude boundary
    DM dma;
    PetscCall(DMAdaptLabel(dm, marker, &dma));
    PetscCall(DMViewFromOptions(dma, NULL, "-adapt_dm_view"));

    // clean up
    PetscCall(DMDestroy(&dm));
    PetscCall(DMDestroy(&surface));
    PetscCall(DMDestroy(&dma));
    PetscCall(PetscFinalize());
    return 0;
}

/*TEST

    args: -dm_adaptor cellrefiner -dm_plex_transform_type extrude -dm_view -sur_dm_view -adapt_dm_view \
    -dm_plex_transform_extrude_thickness 0.5 -dm_plex_simplex 0  -dm_plex_transform_extrude_use_tensor 0


TEST*/
