#ifndef ABLATELIBRARY_LSVOF_HPP
#define ABLATELIBRARY_LSVOF_HPP

void VOF_1D(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceLength, PetscReal *cellLength);

void VOF_2D_Tri(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceLength, PetscReal *cellArea);
void VOF_2D_Quad(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceLength, PetscReal *cellArea);

void VOF_3D_Tetra(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceArea, PetscReal *cellVol);
void VOF_3D_Hex(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceArea, PetscReal *cellVol);

#endif  // ABLATELIBRARY_LSVOF_HPP
