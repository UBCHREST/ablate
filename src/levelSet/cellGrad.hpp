PetscErrorCode Grad_1D(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g);

PetscErrorCode Grad_2D_Tri(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g);
PetscErrorCode Grad_2D_Quad(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g);

PetscErrorCode Grad_3D_Tetra(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g);
PetscErrorCode Grad_3D_Hex(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g);
