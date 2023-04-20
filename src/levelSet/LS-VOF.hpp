#ifndef ABLATELIBRARY_LSVOF_HPP
#define ABLATELIBRARY_LSVOF_HPP

void VOF_1D(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceLength, PetscReal *cellLength);

void VOF_2D_Tri(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceLength, PetscReal *cellArea);
void VOF_2D_Quad(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceLength, PetscReal *cellArea);


void VOF_3D_Tetra(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceArea, PetscReal *cellVol);
void VOF_3D_Hex(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceArea, PetscReal *cellVol);



PetscErrorCode Grad_1D(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g);

PetscErrorCode Grad_2D_Tri(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g);
PetscErrorCode Grad_2D_Quad(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g);

PetscErrorCode Grad_3D_Tetra(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g);
PetscErrorCode Grad_3D_Hex(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g);


#endif  // ABLATELIBRARY_LSVOF_HPP
