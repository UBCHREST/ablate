void VOF_2D_Tri(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceLength, PetscReal *cellArea);
void VOF_2D_Quad(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceLength, PetscReal *cellArea);


void VOF_3D_Tetra(const PetscReal coords[12], const PetscReal c[4], PetscReal *vof, PetscReal *faceArea, PetscReal *cellVol);
void VOF_3D_Hex(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceArea, PetscReal *cellVol);


