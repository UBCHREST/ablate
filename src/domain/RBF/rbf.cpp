#include "rbf.hpp"
#include <petsc/private/dmpleximpl.h>
#include "utilities/petscSupport.hpp"

using namespace ablate::domain::rbf;

// Return an array of length-3 containing the location, even for 1D or 2D domains
void RBF::Loc3D(PetscInt dim, PetscReal xIn[], PetscReal x[3]) {
    PetscInt d;

    for (d = 0; d < dim; ++d) {
        x[d] = xIn[d];
    }
    for (d = dim; d < 3; ++d) {
        x[d] = 0.0;
    }
}

// Distance between two points
PetscReal RBF::DistanceSquared(PetscInt dim, PetscReal x[], PetscReal y[]) {
    PetscReal r = 0.0;
    for (PetscInt d = 0; d < dim; ++d) {
        r += (x[d] - y[d]) * (x[d] - y[d]);
    }

    return r;
}

// Distance between point and the origin
PetscReal RBF::DistanceSquared(PetscInt dim, PetscReal x[]) {
    PetscReal r = 0.0;
    for (PetscInt d = 0; d < dim; ++d) {
        r += x[d] * x[d];
    }
    return r;
}

#include <slepcsvd.h>
PetscErrorCode ConditionNumber(Mat A) {
    SVD svd; /* singular value solver context */
    PetscReal sigma_1, sigma_n;
    PetscInt nconv1, nconv2;

    PetscFunctionBeginUser;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
               Create the singular value solver and set the solution method
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Create singular value context
    */
    PetscCall(SVDCreate(PETSC_COMM_WORLD, &svd));

    /*
       Set operator
    */
    PetscCall(SVDSetOperators(svd, A, NULL));

    /*
      Set the method to use
    */
    PetscCall(SVDSetType(svd, SVDLAPACK));

    /*
       Set solver parameters at runtime
    */
    PetscCall(SVDSetFromOptions(svd));
    PetscCall(SVDSetDimensions(svd, 1, PETSC_DEFAULT, PETSC_DEFAULT));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                        Solve the singular value problem
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       First request a singular value from one end of the spectrum
    */
    PetscCall(SVDSetWhichSingularTriplets(svd, SVD_LARGEST));
    PetscCall(SVDSolve(svd));
    /*
       Get number of converged singular values
    */
    PetscCall(SVDGetConverged(svd, &nconv1));
    /*
       Get converged singular values: largest singular value is stored in sigma_1.
       In this example, we are not interested in the singular vectors
    */
    if (nconv1 > 0)
        PetscCall(SVDGetSingularTriplet(svd, 0, &sigma_1, NULL, NULL));
    else
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Unable to compute large singular value!\n\n"));

    /*
       Request a singular value from the other end of the spectrum
    */
    PetscCall(SVDSetWhichSingularTriplets(svd, SVD_SMALLEST));
    PetscCall(SVDSolve(svd));
    /*
       Get number of converged singular triplets
    */
    PetscCall(SVDGetConverged(svd, &nconv2));
    /*
       Get converged singular values: smallest singular value is stored in sigma_n.
       As before, we are not interested in the singular vectors
    */
    if (nconv2 > 0)
        PetscCall(SVDGetSingularTriplet(svd, 0, &sigma_n, NULL, NULL));
    else
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Unable to compute small singular value!\n\n"));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Display solution and clean up
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    if (nconv1 > 0 && nconv2 > 0) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Computed singular values: sigma_1=%.4e, sigma_n=%.4e\n", (double)sigma_1, (double)sigma_n));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, " Estimated condition number: sigma_1/sigma_n=%.4e\n\n", (double)(sigma_1 / sigma_n)));
    }

    /*
       Free work space
    */
    PetscCall(SVDDestroy(&svd));

    return 0;
}

static PetscInt fac[11] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800};  // Pre-computed factorials

// Compute the LU-factorization of the augmented RBF matrix
// c - Location in cellRange
void RBF::Matrix(const PetscInt c) {
    PetscInt i, j, d, px, py, pz, matSize;
    PetscInt nCells, *list;
    const PetscInt dim = RBF::subDomain->GetDimensions();
    const PetscInt numPoly = RBF::nPoly, p = RBF::polyOrder, p1 = PetscMax(p + 1, 1);
    Mat A;
    PetscReal *x, *vals;
    PetscReal x0[dim];  // Center of the cell of interest
    PetscReal *xp;      // Powers of the cell centers
    const DM dm = RBF::subDomain->GetSubDM();

    // Get the list of neighbor cells
    DMPlexGetNeighbors(dm, c, -1, -1.0, RBF::minNumberCells, RBF::useCells, (PetscBool)(RBF::returnNeighborVertices), &nCells, &list) >> utilities::PetscUtilities::checkError;

    if (numPoly >= nCells) {
        throw std::invalid_argument("Number of surrounding cells, " + std::to_string(nCells) + ", can not support a requested polynomial order of " + std::to_string(p) + " which requires " +
                                    std::to_string(numPoly) + " number of cells.");
    }

    PetscMalloc1(nCells * dim * p1, &xp) >> utilities::PetscUtilities::checkError;

    // Get the cell center
    DMPlexComputeCellGeometryFVM(dm, c, NULL, x0, NULL) >> utilities::PetscUtilities::checkError;

    // Shifted cell-centers of neighbor cells
    PetscMalloc1(nCells * dim, &x);
    for (i = 0; i < nCells; ++i) {
        DMPlexComputeCellGeometryFVM(dm, list[i], NULL, &x[i * dim], NULL) >> utilities::PetscUtilities::checkError;
        for (d = 0; d < dim; ++d) {
            x[i * dim + d] -= x0[d];

            // Precompute the powers for later use
            xp[(i * dim + d) * p1 + 0] = 1.0;
            for (px = 1; px < p1; ++px) {
                xp[(i * dim + d) * p1 + px] = xp[(i * dim + d) * p1 + (px - 1)] * x[i * dim + d];
            }
        }
    }

    matSize = nCells + numPoly;

    // Create the matrix
    MatCreateSeqDense(PETSC_COMM_SELF, matSize, matSize, NULL, &A) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)A, "ablate::domain::RBF::A") >> utilities::PetscUtilities::checkError;
    MatZeroEntries(A) >> utilities::PetscUtilities::checkError;
    MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE) >> utilities::PetscUtilities::checkError;

    MatDenseGetArrayWrite(A, &vals) >> utilities::PetscUtilities::checkError;

    // RBF contributions to the matrix
    for (i = 0; i < nCells; ++i) {
        for (j = i; j < nCells; ++j) {
            vals[i * matSize + j] = vals[j * matSize + i] = RBFVal(dim, &x[i * dim], &x[j * dim]);
        }
    }

    if (numPoly > 0) {
        // Augmented polynomial contributions
        switch (dim) {
            case 1:
                for (i = 0; i < nCells; ++i) {
                    j = nCells;
                    for (px = 0; px < p1; ++px) {
                        vals[i * matSize + j] = vals[j * matSize + i] = xp[(i * dim + 0) * p1 + px];
                        ++j;
                    }
                }
                break;
            case 2:
                for (i = 0; i < nCells; ++i) {
                    j = nCells;
                    for (py = 0; py < p1; ++py) {
                        for (px = 0; px < p1 - py; ++px) {
                            vals[i * matSize + j] = vals[j * matSize + i] = xp[(i * dim + 0) * p1 + px] * xp[(i * dim + 1) * p1 + py];
                            ++j;
                        }
                    }
                }
                break;
            case 3:
                for (i = 0; i < nCells; ++i) {
                    j = nCells;
                    for (pz = 0; pz < p1; ++pz) {
                        for (py = 0; py < p1 - pz; ++py) {
                            for (px = 0; px < p1 - py - pz; ++px) {
                                vals[i * matSize + j] = vals[j * matSize + i] = xp[(i * dim + 0) * p1 + px] * xp[(i * dim + 1) * p1 + py] * xp[(i * dim + 2) * p1 + pz];
                                ++j;
                            }
                        }
                    }
                }
                break;
            default:
                throw std::runtime_error("ablate::domain::RBF::Matrix encountered an unknown dimension.");
        }
    }

    MatDenseRestoreArrayWrite(A, &vals) >> utilities::PetscUtilities::checkError;
    MatViewFromOptions(A, NULL, "-ablate::domain::rbf::RBF::A_view") >> utilities::PetscUtilities::checkError;

    // Factor the matrix
    MatLUFactor(A, NULL, NULL, NULL) >> utilities::PetscUtilities::checkError;

    PetscFree(xp) >> utilities::PetscUtilities::checkError;

    // Assign output
    RBF::RBFMatrix[c] = A;
    RBF::stencilXLocs[c] = x;
    RBF::nStencil[c] = nCells;
    PetscMalloc1(nCells, &(RBF::stencilList[c])) >> utilities::PetscUtilities::checkError;
    PetscArraycpy(RBF::stencilList[c], list, nCells) >> utilities::PetscUtilities::checkError;

    // Return the work arrays
    DMPlexRestoreNeighbors(dm, c, -1, -1.0, RBF::minNumberCells, RBF::useCells, (PetscBool)(RBF::returnNeighborVertices), &nCells, &list) >> utilities::PetscUtilities::checkError;
}

/************ Begin Derivative Code **********************/

void RBF::SetDerivatives(PetscInt numDer, PetscInt dx[], PetscInt dy[], PetscInt dz[], PetscBool useCellsLocal) {
    if (numDer > 0) {
        RBF::useCells = useCellsLocal;
        RBF::nDer = numDer;

        PetscMalloc1(3 * numDer, &(RBF::dxyz)) >> utilities::PetscUtilities::checkError;
        PetscHMapICreate(&(RBF::hash)) >> utilities::PetscUtilities::checkError;

        // Store the derivatives
        for (PetscInt i = 0; i < numDer; ++i) {
            RBF::dxyz[i * 3 + 0] = dx[i];
            RBF::dxyz[i * 3 + 1] = dy[i];
            RBF::dxyz[i * 3 + 2] = dz[i];
            PetscHMapISet(RBF::hash, RBF::derivativeKey(dx[i], dy[i], dz[i]), i);
        }
    }
}

/**
 * Set derivatives, defaulting to using vertices
 */
void RBF::SetDerivatives(PetscInt numDer, PetscInt dx[], PetscInt dy[], PetscInt dz[]) { RBF::SetDerivatives(numDer, dx, dy, dz, PETSC_FALSE); }

// Compute the RBF weights at the cell center of p using a cell-list
// c - The center cell in cellRange ordering
void RBF::SetupDerivativeStencils(PetscInt c) {
    if (RBF::RBFMatrix[c] == nullptr) {
        RBF::Matrix(c);
    }

    Mat A = RBF::RBFMatrix[c], B = nullptr;
    PetscReal *x = RBF::stencilXLocs[c];  // Shifted cell-centers of the neigbor list
    PetscInt dim = RBF::subDomain->GetDimensions();
    PetscInt nCells = RBF::nStencil[c];
    PetscInt matSize = nCells + RBF::nPoly;
    PetscInt numDer = RBF::nDer;
    PetscInt *derXYZ = RBF::dxyz;
    PetscInt i, j;
    PetscInt px, py, pz, p1 = PetscMax(RBF::polyOrder + 1, 1);
    PetscScalar *vals = nullptr;

    // Create the RHS
    MatCreateSeqDense(PETSC_COMM_SELF, matSize, numDer, NULL, &B) >> utilities::PetscUtilities::checkError;
    PetscObjectSetName((PetscObject)B, "ablate::domain::rbf::RBF::rhs") >> utilities::PetscUtilities::checkError;
    MatZeroEntries(B) >> utilities::PetscUtilities::checkError;
    MatDenseGetArrayWrite(B, &vals) >> utilities::PetscUtilities::checkError;

    // Derivatives of the RBF
    for (i = 0; i < nCells; ++i) {
        for (j = 0; j < numDer; ++j) {
            vals[i + j * matSize] = RBFDer(dim, &x[i * dim], derXYZ[j * 3 + 0], derXYZ[j * 3 + 1], derXYZ[j * 3 + 2]);
        }
    }

    if (RBF::nPoly > 0) {
        // Derivatives of the augmented polynomials
        switch (dim) {
            case 1:
                for (j = 0; j < numDer; ++j) {
                    i = nCells;
                    for (px = 0; px < p1; ++px) {
                        if (derXYZ[j * 3 + 0] == px) {
                            vals[i + j * matSize] = (PetscReal)fac[px];
                        }
                        ++i;
                    }
                }
                break;
            case 2:
                for (j = 0; j < numDer; ++j) {
                    i = nCells;
                    for (py = 0; py < p1; ++py) {
                        for (px = 0; px < p1 - py; ++px) {
                            if (derXYZ[j * 3 + 0] == px && derXYZ[j * 3 + 1] == py) {
                                vals[i + j * matSize] = (PetscReal)(fac[px] * fac[py]);
                            }
                            ++i;
                        }
                    }
                }
                break;
            case 3:
                for (j = 0; j < numDer; ++j) {
                    i = nCells;
                    for (pz = 0; pz < p1; ++pz) {
                        for (py = 0; py < p1 - pz; ++py) {
                            for (px = 0; px < p1 - py - pz; ++px) {
                                if (derXYZ[j * 3 + 0] == px && derXYZ[j * 3 + 1] == py && derXYZ[j * 3 + 2] == pz) {
                                    vals[i + j * matSize] = (PetscReal)(fac[px] * fac[py] * fac[pz]);
                                }
                                ++i;
                            }
                        }
                    }
                }
                break;
            default:
                throw std::runtime_error("ablate::domain::RBF::SetupDerivativeStencils encountered an unknown dimension.");
        }
    }

    MatDenseRestoreArrayWrite(B, &vals) >> utilities::PetscUtilities::checkError;

    MatViewFromOptions(B, NULL, "-ablate::domain::rbf::RBF::rhs_view") >> utilities::PetscUtilities::checkError;

    MatMatSolve(A, B, B) >> utilities::PetscUtilities::checkError;

    MatViewFromOptions(B, NULL, "-ablate::domain::rbf::RBF::sol_view") >> utilities::PetscUtilities::checkError;

    // Now populate the output
    PetscMalloc1(numDer * nCells, &(RBF::stencilWeights[c])) >> utilities::PetscUtilities::checkError;
    PetscReal *wt = RBF::stencilWeights[c];
    MatDenseGetArrayWrite(B, &vals) >> utilities::PetscUtilities::checkError;
    for (i = 0; i < nCells; ++i) {
        for (j = 0; j < numDer; ++j) {
            wt[i * numDer + j] = vals[i + j * matSize];
        }
    }
    MatDenseGetArrayWrite(B, &vals) >> utilities::PetscUtilities::checkError;

    // The matrix is only needed in the future if interpolation is required. If not then
    //    the stencil list is enough.
    if (RBF::hasInterpolation) {
        RBF::RBFMatrix[c] = A;
        RBF::stencilXLocs[c] = x;
    } else {
        MatDestroy(&A) >> utilities::PetscUtilities::checkError;
        PetscFree(x) >> utilities::PetscUtilities::checkError;
    }
    MatDestroy(&B) >> utilities::PetscUtilities::checkError;
}

/*
 * Setup all derivative stencils for the entire subDomain
 */
void RBF::SetupDerivativeStencils() {
    for (PetscInt c = cStart; c < cEnd; ++c) {
        RBF::SetupDerivativeStencils(RBF::cellList[c]);
    }
}

PetscReal RBF::EvalDer(const ablate::domain::Field *field, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz) {
    RBF::CheckField(field);

    return RBF::EvalDer(RBF::subDomain->GetFieldDM(*field), RBF::subDomain->GetVec(*field), field->id, c, dx, dy, dz);
}

PetscReal RBF::EvalDer(DM dm, Vec vec, const PetscInt fid, PetscInt c, PetscInt dx, PetscInt dy, PetscInt dz) {

    PetscReal *wt = nullptr;
    PetscScalar val = 0.0, *f;
    const PetscScalar *array;
    PetscInt nCells = -1, *lst = nullptr;
    PetscInt derID = -1, numDer = RBF::nDer;

    PetscBool hasKey;
    PetscInt derKey = RBF::derivativeKey(dx, dy, dz);
    PetscHMapIHas(RBF::hash, derKey, &hasKey);
    if (!hasKey) throw std::invalid_argument("RBF: Derivative of (" + std::to_string(dx) + ", " + std::to_string(dy) + ", " + std::to_string(dz) + ") is not setup.");
    PetscHMapIGet(RBF::hash, derKey, &derID);

    // If the stencil hasn't been setup yet do so
    if (RBF::nStencil[c] < 1) {
        RBF::SetupDerivativeStencils(c);
    }

    wt = RBF::stencilWeights[c];
    nCells = RBF::nStencil[c];
    lst = RBF::stencilList[c];

    VecGetArrayRead(vec, &array) >> utilities::PetscUtilities::checkError;

    for (PetscInt i = 0; i < nCells; ++i) {
        if (fid >= 0) {
            DMPlexPointLocalFieldRead(dm, lst[i], fid, array, &f) >> utilities::PetscUtilities::checkError;
        } else {
            DMPlexPointLocalRead(dm, lst[i], array, &f) >> utilities::PetscUtilities::checkError;
        }

        val += wt[i * numDer + derID] * (*f);
    }

    VecRestoreArrayRead(vec, &array) >> utilities::PetscUtilities::checkError;

    return val;
}
/************ End Derivative Code **********************/

/************ Begin Interpolation Code **********************/

PetscReal RBF::Interpolate(const ablate::domain::Field *field, PetscReal xEval[3]) {
    RBF::CheckField(field);

    return RBF::Interpolate(field, RBF::subDomain->GetVec(*field), xEval);
}

PetscReal RBF::Interpolate(const ablate::domain::Field *field, Vec f, PetscReal xEval[3]) {
    PetscInt i, c, nCells, *lst;
    PetscScalar *vals, *v;
    const PetscScalar *fvals;
    PetscReal *x, x0[3];
    Mat A;
    Vec weights, rhs;
    DM dm = RBF::subDomain->GetFieldDM(*field);
    const PetscInt fid = field->id;

    DMPlexGetContainingCell(dm, xEval, &c) >> utilities::PetscUtilities::checkError;

    if (c < 0) {
        throw std::runtime_error("ablate::domain::RBF::Interpolate could not determine the location of (" + std::to_string(xEval[0]) + ", " + std::to_string(xEval[1]) + ", " +
                                 std::to_string(xEval[2]) + ").");
    }

    if (RBF::RBFMatrix[c] == nullptr) {
        RBF::Matrix(c);
    }

    A = RBF::RBFMatrix[c];
    x = RBF::stencilXLocs[c];

    nCells = RBF::nStencil[c];
    lst = RBF::stencilList[c];

    MatCreateVecs(A, &weights, &rhs) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(weights) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(rhs) >> utilities::PetscUtilities::checkError;

    // The function values
    VecGetArrayRead(f, &fvals) >> utilities::PetscUtilities::checkError;
    VecGetArray(rhs, &vals) >> utilities::PetscUtilities::checkError;

    for (i = 0; i < nCells; ++i) {
        // DMPlexPointLocalFieldRead isn't behaving like I would expect. If I don't make f a pointer then it just returns zero.
        //    Additionally, it looks like it allows for the editing of the value.
        if (fid >= 0) {
            DMPlexPointLocalFieldRead(dm, lst[i], fid, fvals, &v) >> utilities::PetscUtilities::checkError;
        } else {
            DMPlexPointLocalRead(dm, lst[i], fvals, &v) >> utilities::PetscUtilities::checkError;
        }

        vals[i] = *v;
    }

    VecRestoreArrayRead(f, &fvals) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(rhs, &vals) >> utilities::PetscUtilities::checkError;

    MatSolve(A, rhs, weights) >> utilities::PetscUtilities::checkError;

    VecDestroy(&rhs) >> utilities::PetscUtilities::checkError;

    // Now do the actual interpolation

    // Get the cell center
    DMPlexComputeCellGeometryFVM(dm, c, NULL, x0, NULL) >> utilities::PetscUtilities::checkError;

    PetscInt p1 = PetscMax(RBF::polyOrder + 1, 1), dim = subDomain->GetDimensions();
    PetscInt px, py, pz;
    PetscReal *xp;
    PetscMalloc1(dim * p1, &xp) >> utilities::PetscUtilities::checkError;

    for (PetscInt d = 0; d < dim; ++d) {
        x0[d] = xEval[d] - x0[d];  // Shifted center

        // precompute powers
        xp[d * p1 + 0] = 1.0;
        for (px = 1; px < p1; ++px) {
            xp[d * p1 + px] = xp[d * p1 + (px - 1)] * x0[d];
        }
    }

    PetscReal interpVal = 0.0;
    VecGetArray(weights, &vals) >> utilities::PetscUtilities::checkError;
    for (i = 0; i < nCells; ++i) {
        interpVal += vals[i] * RBFVal(dim, x0, &x[i * dim]);
    }

    // Augmented polynomial contributions
    switch (dim) {
        case 1:
            for (px = 0; px < p1; ++px) {
                interpVal += vals[i++] * xp[0 * p1 + px];
            }
            break;
        case 2:
            for (py = 0; py < p1; ++py) {
                for (px = 0; px < p1 - py; ++px) {
                    interpVal += vals[i++] * xp[0 * p1 + px] * xp[1 * p1 + py];
                }
            }
            break;
        case 3:
            for (pz = 0; pz < p1; ++pz) {
                for (py = 0; py < p1 - pz; ++py) {
                    for (px = 0; px < p1 - py - pz; ++px) {
                        interpVal += vals[i++] * xp[0 * p1 + px] * xp[1 * p1 + py] * xp[2 * p1 + pz];
                    }
                }
            }
            break;
        default:
            throw std::runtime_error("ablate::domain::RBF::Interpolate encountered an unknown dimension.");
    }

    VecRestoreArray(weights, &vals) >> utilities::PetscUtilities::checkError;
    VecDestroy(&weights) >> utilities::PetscUtilities::checkError;
    PetscFree(xp) >> utilities::PetscUtilities::checkError;

    return interpVal;
}

/************ End Interpolation Code **********************/

/************ Constructor, Setup, and Initialization Code **********************/
RBF::RBF(int polyOrder, bool hasDerivatives, bool hasInterpolation, bool returnNeighborVertices)
    : polyOrder(polyOrder), returnNeighborVertices(returnNeighborVertices), hasDerivatives(hasDerivatives), hasInterpolation(hasInterpolation) {}

RBF::~RBF() {
    RBF::FreeStencilData();

    if (dxyz) {
        PetscFree(dxyz);
    }
    if (hash) {
        PetscHMapIDestroy(&hash);
    }
}

void RBF::FreeStencilData() {
    if ((RBF::cEnd - RBF::cStart) > 0) {
        for (PetscInt c = RBF::cStart; c < RBF::cEnd; ++c) {
            PetscFree(RBF::stencilList[c]);
            if (RBF::RBFMatrix[c]) MatDestroy(&(RBF::RBFMatrix[c]));
            PetscFree(RBF::stencilWeights[c]);
            PetscFree(RBF::stencilXLocs[c]);
        }
        RBF::cellList += cStart;
        RBF::nStencil += cStart;
        RBF::stencilList += cStart;
        RBF::RBFMatrix += cStart;
        RBF::stencilXLocs += cStart;
        RBF::stencilWeights += cStart;
        PetscFree6(RBF::cellList, RBF::nStencil, RBF::stencilList, RBF::RBFMatrix, RBF::stencilXLocs, RBF::stencilWeights) >> utilities::PetscUtilities::checkError;
    }
}

void RBF::CheckField(const ablate::domain::Field *field) {  // Checks whether the field is SOL or AUX
    PetscMPIInt size;
    DM dm = RBF::subDomain->GetFieldDM(*field);

    MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size);
    if (field->location == FieldLocation::SOL && size > 1) {
        // Only global vectors of SOL fields are stored in the subDomain. The subDomain needs to be augmented to also store the local vector of SOL fields
        //  before this function can work on them. One work around is to copy the SOL field into a dummy AUX field and work on that.
        throw std::runtime_error("ablate::domain::RBF does not work on SOL fields in parallel. A local vector needs to be obtained first.");
    }
}

// This is done once
void RBF::Setup(std::shared_ptr<ablate::domain::SubDomain> subDomainIn) {
    if ((!RBF::hasDerivatives) && (!RBF::hasInterpolation)) {
        throw std::runtime_error("ablate::domain::RBF requires either derivatives or interpolation.");
    }

    RBF::subDomain = subDomainIn;

    PetscInt dim = subDomainIn->GetDimensions();

    //   The number of polynomial values is (p+2)(p+1)/2 in 2D and (p+3)(p+2)(p+1)/6 in 3D
    PetscInt p = RBF::polyOrder;

    // Set the size of the augmented polynomial. A value of p<0 means no augmented polynomials.
    if (p < 0) {
        RBF::nPoly = 0;
    } else if (dim == 1) {
        RBF::nPoly = p + 1;
    } else if (dim == 2) {
        RBF::nPoly = (p + 2) * (p + 1) / 2;
    } else {
        RBF::nPoly = (p + 3) * (p + 2) * (p + 1) / 6;
    }

    if (p < 0) {
        // If there isn't an augmenting polynomial (not recommended) then set the number of point to 5^dim.
        RBF::minNumberCells = PetscPowInt(5, dim);
    } else {
        // Set the minimum number of cells to get compute the RBF matrix
        // The unisolvency requirement is that the number of points to use be larger than twice the number of elements in the augmenting polynomial.
        RBF::minNumberCells = (PetscInt)floor(2 * (RBF::nPoly));
    }

    if (RBF::hasDerivatives) {
        // Now setup the derivatives required for curvature/normal calculations. This should probably move over to user-option
        PetscInt numDer = 0;
        PetscInt dx[10], dy[10], dz[10];

        dx[numDer] = 0;
        dy[numDer] = 0;
        dz[numDer++] = 0;

        if (dim >= 1) {
            dx[numDer] = 1;
            dy[numDer] = 0;
            dz[numDer++] = 0;
            dx[numDer] = 2;
            dy[numDer] = 0;
            dz[numDer++] = 0;
        }

        if (dim >= 2) {
            dx[numDer] = 0;
            dy[numDer] = 1;
            dz[numDer++] = 0;
            dx[numDer] = 1;
            dy[numDer] = 1;
            dz[numDer++] = 0;
            dx[numDer] = 0;
            dy[numDer] = 2;
            dz[numDer++] = 0;
        }

        if (dim == 3) {
            dx[numDer] = 0;
            dy[numDer] = 0;
            dz[numDer++] = 1;
            dx[numDer] = 1;
            dy[numDer] = 0;
            dz[numDer++] = 1;
            dx[numDer] = 0;
            dy[numDer] = 1;
            dz[numDer++] = 1;
            dx[numDer] = 0;
            dy[numDer] = 0;
            dz[numDer++] = 2;
        }

        SetDerivatives(numDer, dx, dy, dz, PETSC_FALSE);
    }
}

void RBF::Initialize() {
    ablate::domain::Range range;

    // Grab the range of cells from the subDomain
    RBF::subDomain->GetCellRange(nullptr, range);

    // If this is called due to a grid change then release the old memory. In this case cEnd - cStart will be greater than zero.
    RBF::FreeStencilData();

    RBF::cStart = range.start;
    RBF::cEnd = range.end;

    // Both interpolation and derivatives need the list of points
    PetscInt nCells = RBF::cEnd - RBF::cStart;
    PetscMalloc6(nCells, &(RBF::cellList), nCells, &(RBF::nStencil), nCells, &(RBF::stencilList), nCells, &(RBF::RBFMatrix), nCells, &(RBF::stencilXLocs), nCells, &(RBF::stencilWeights)) >>
        utilities::PetscUtilities::checkError;

    // Shift so that we can use cell range directly
    RBF::cellList -= cStart;
    RBF::nStencil -= cStart;
    RBF::stencilList -= cStart;
    RBF::RBFMatrix -= cStart;
    RBF::stencilXLocs -= cStart;
    RBF::stencilWeights -= cStart;

    for (PetscInt c = cStart; c < cEnd; ++c) {
        RBF::cellList[c] = range.GetPoint(c);

        RBF::nStencil[c] = -1;
        RBF::stencilList[c] = nullptr;

        RBF::RBFMatrix[c] = nullptr;
        RBF::stencilXLocs[c] = nullptr;
        RBF::stencilWeights[c] = nullptr;
    }

    subDomain->RestoreRange(range);
}
