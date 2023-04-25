PetscErrorCode DMPlexGetNumCellVertices(DM dm, const PetscInt p, PetscInt *nv);

PetscErrorCode DMPlexGetCellVertices(DM dm, const PetscInt p, PetscInt *nVerts, PetscInt *verts[]);
PetscErrorCode DMPlexRestoreCellVertices(DM dm, const PetscInt p, PetscInt *nVerts, PetscInt *vertOut[]);

PetscErrorCode DMPlexGetVertexCoordinates(DM dm, const PetscInt np, const PetscInt pArray[], PetscScalar *coords[]);
PetscErrorCode DMPlexRestoreVertexCoordinates(DM dm, const PetscInt np, const PetscInt pArray[], PetscScalar *coords[]);
