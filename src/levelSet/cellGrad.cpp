#include <petsc.h>
#include <petscblaslapack.h>

// Note about the gradient calculations: These use a Taylor-Series expansion about the point-of-interest.
//  For triangles(2D) or tets(3D) this is a well-defined linear system.
//  For quads(2D) or hexes(3D) this is an over-constrained system.
//  Another possible method would be to use the Green-Gauss Gradient (GGG) method which relates the (constant) gradient in a cell
//  to an integral around the cell edges/faces. From cursory 2D investigations of triangles and quads the error between the two
//  methods is comparable and both have first-order convergence. The GGG method is simple in 2D but becomes more complicated in 3D.
//  This is why we're using the Taylor-Series version here, even though there is a (small) linear system that needs to be solved each time.

/**
 * Cell-wise function value and gradient at a given location for 1D element
 * @param x0 - Location to compute gradient and function value
 * @param coords - Vertex locations
 * @param c - Function to find gradient of at cell center
 * @param c0 - The function value at x0
 * @param g - The gradient at x0
 */
PetscErrorCode Grad_1D(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g) {
    PetscFunctionBegin;

    if (c0) *c0 = c[0] + (c[1] - c[0]) * (x0[0] - coords[0]) / (coords[1] - coords[0]);
    if (g) g[0] = (c[1] - c[0]) / (coords[1] - coords[0]);

    PetscFunctionReturn(0);
}

/**
 * Cell-wise function value and gradient at a given location for triangles
 * @param x0 - Location to compute gradient and function value
 * @param coords - Vertex locations
 * @param c - Function to find gradient of at cell center
 * @param c0 - The function value at x0
 * @param g - The gradient at x0
 */
PetscErrorCode Grad_2D_Tri(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g) {
    // Obtains the function value and gradient at a point in a cell given information on the cell vertices using a Taylor-Series expansion.

    PetscReal A[9] = {1.0, 1.0, 1.0, coords[0] - x0[0], coords[2] - x0[0], coords[4] - x0[0], coords[1] - x0[1], coords[3] - x0[1], coords[5] - x0[1]};
    PetscReal x[3] = {c[0], c[1], c[2]};
    PetscBLASInt info, nrhs = 1, ipiv[3], n = 3;

    PetscFunctionBegin;

    PetscCallBLAS("LAPACKgesv", LAPACKgesv_(&n, &nrhs, A, &n, ipiv, x, &n, &info));
    PetscCheck(!info, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error while calling the Lapack routine DGESV");

    if (c0) *c0 = x[0];
    if (g) {
        g[0] = x[1];
        g[1] = x[2];
    }

    PetscFunctionReturn(0);
}

/**
 * Cell-wise function value and gradient at a given location for quadrilaterals
 * @param x0 - Location to compute gradient and function value
 * @param coords - Vertex locations
 * @param c - Function to find gradient of at cell center
 * @param c0 - The function value at x0
 * @param g - The gradient at x0
 */
PetscErrorCode Grad_2D_Quad(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g) {
    // Obtains the function value and gradient at a point in a cell given information on the cell vertices using a Taylor-Series expansion.

    PetscReal A[12] = {1.0, 1.0, 1.0, 1.0, coords[0] - x0[0], coords[2] - x0[0], coords[4] - x0[0], coords[6] - x0[0], coords[1] - x0[1], coords[3] - x0[1], coords[5] - x0[1], coords[7] - x0[1]};
    PetscReal x[4] = {c[0], c[1], c[2], c[3]};
    char transpose = 'N';
    PetscBLASInt m = 4, n = 3, nrhs = 1, info, worksize = m * n;
    PetscReal work[m * n];

    PetscFunctionBegin;

    PetscCallBLAS("LAPACKgels", LAPACKgels_(&transpose, &m, &n, &nrhs, A, &m, x, &m, work, &worksize, &info));
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Bad argument to GELS");

    if (c0) *c0 = x[0];
    if (g) {
        g[0] = x[1];
        g[1] = x[2];
    }

    PetscFunctionReturn(0);
}

/**
 * Cell-wise function value and gradient at a given location for tetrahedrals
 * @param x0 - Location to compute gradient and function value
 * @param coords - Vertex locations
 * @param c - Function to find gradient of at cell center
 * @param c0 - The function value at x0
 * @param g - The gradient at x0
 */
PetscErrorCode Grad_3D_Tetra(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g) {
    // Obtains the function value and gradient at a point in a cell given information on the cell vertices using a Taylor-Series expansion.

    PetscReal A[16] = {1.0,
                       1.0,
                       1.0,
                       1.0,
                       coords[0] - x0[0],
                       coords[3] - x0[0],
                       coords[6] - x0[0],
                       coords[9] - x0[0],
                       coords[1] - x0[1],
                       coords[4] - x0[1],
                       coords[7] - x0[1],
                       coords[10] - x0[1],
                       coords[2] - x0[2],
                       coords[5] - x0[2],
                       coords[8] - x0[2],
                       coords[11] - x0[2]};
    PetscReal x[4] = {c[0], c[1], c[2], c[3]};
    PetscBLASInt info, nrhs = 1, ipiv[4], n = 4;

    PetscFunctionBegin;

    PetscCallBLAS("LAPACKgesv", LAPACKgesv_(&n, &nrhs, A, &n, ipiv, x, &n, &info));
    PetscCheck(!info, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error while calling the Lapack routine DGESV");

    if (c0) *c0 = x[0];
    if (g) {
        g[0] = x[1];
        g[1] = x[2];
        g[2] = x[3];
    }

    PetscFunctionReturn(0);
}

/**
 * Cell-wise function value and gradient at a given location for hexagons
 * @param x0 - Location to compute gradient and function value
 * @param coords - Vertex locations
 * @param c - Function to find gradient of at cell center
 * @param c0 - The function value at x0
 * @param g - The gradient at x0
 */
PetscErrorCode Grad_3D_Hex(const PetscReal x0[], const PetscReal coords[], const PetscReal c[], PetscReal *c0, PetscReal *g) {
    // Obtains the function value and gradient at a point in a cell given information on the cell vertices using a Taylor-Series expansion.

    PetscReal A[32], x[8];
    char transpose = 'N';
    PetscBLASInt m = 8, n = 4, nrhs = 1, info, worksize = m * n;
    PetscReal work[m * n];

    PetscFunctionBegin;

    for (PetscInt i = 0; i < 8; ++i) {
        x[i] = c[i];
        A[i] = 1.0;
        A[i + 8] = coords[i * 3 + 0] - x0[0];
        A[i + 16] = coords[i * 3 + 1] - x0[1];
        A[i + 24] = coords[i * 3 + 2] - x0[2];
    }

    PetscCallBLAS("LAPACKgels", LAPACKgels_(&transpose, &m, &n, &nrhs, A, &m, x, &m, work, &worksize, &info));
    PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Bad argument to GELS");

    if (c0) *c0 = x[0];
    if (g) {
        g[0] = x[1];
        g[1] = x[2];
        g[2] = x[3];
    }

    PetscFunctionReturn(0);
}
