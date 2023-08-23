#include <petsc.h>

// Area of a triangle.
// It's 1/2 of the determinant of
//   | x1 x2 |
//   | y1 y2 |
// where the coordinates are shifted so that x[0] is at the origin.
static PetscReal CellArea_Triangle(const PetscReal coords[]) {
    const PetscReal x1 = coords[2] - coords[0], y1 = coords[3] - coords[1];
    const PetscReal x2 = coords[4] - coords[0], y2 = coords[5] - coords[1];
    const PetscReal area = 0.5 * (x1 * y2 - y1 * x2);

    return PetscAbsReal(area);
}

// Volume of a tetrahedron.
// It's 1/6 of the determinant of
//   | x1 x2 x3 |
//   | y1 y2 y3 |
//   | z1 z2 z3 |
// where the coordinates are shifted so that x[0] is at the origin.
static PetscReal CellVolume_Tetrahedron(const PetscReal coords[]) {
    const PetscReal x1 = coords[3] - coords[0], y1 = coords[4] - coords[1], z1 = coords[5] - coords[2];
    const PetscReal x2 = coords[6] - coords[0], y2 = coords[7] - coords[1], z2 = coords[8] - coords[2];
    const PetscReal x3 = coords[9] - coords[0], y3 = coords[10] - coords[1], z3 = coords[11] - coords[2];
    const PetscReal vol = (x2 * y3 * z1 + x3 * y1 * z2 + x1 * y2 * z3 - x3 * y2 * z1 - x2 * y1 * z3 - x1 * y3 * z2) / 6.0;

    return PetscAbsReal(vol);
}

void VOF_1D(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceLength, PetscReal *cellLength) {
    if (faceLength) *faceLength = 0.0;  // Kind of pointless. Want to have the same call as the 2D/3D ones.
    if (cellLength) *cellLength = coords[1] - coords[0];

    if (vof) {
        if (c[0] * c[1] < 0.0) {  // Has an interface crossing
            *vof = PetscMin(c[0], c[1]) / (PetscMin(c[0], c[1]) - PetscMax(c[0], c[1]));
        } else {
            *vof = (c[0] <= 0 ? 1.0 : 0.0);
        }
    }
}

// 2D Simplex: DM_POLYTOPE_TRIANGLE
// Note: The article returns the area of the unit triangle. To get the actual volume you would
//    multiply the value by 2*(volume of the cell). Since we're interested in the VOF we simply multiply
//    the value by 2, as we would be dividing by the volume of the cell to get the VOF.
void VOF_2D_Tri(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceLength, PetscReal *cellArea) {
    if (vof || faceLength) {
        //    PetscReal p[3], x[6];
        PetscInt l[3];
        PetscReal p[2];
        if (c[0] >= 0.0 && c[1] >= 0.0 && c[2] >= 0.0) {
            if (vof) *vof = 0.0;
            if (faceLength) *faceLength = 0.0;
        } else if (c[0] <= 0.0 && c[1] <= 0.0 && c[2] <= 0.0) {
            if (vof) *vof = 1.0;
            if (faceLength) *faceLength = 0.0;
        } else {
            // This orders the nodes so that the vertex with the opposite sign of the other two is always in position 2
            if ((c[0] >= 0.0 && c[1] < 0.0 && c[2] < 0.0) || (c[0] < 0.0 && c[1] >= 0.0 && c[2] >= 0.0)) {
                l[0] = 2;
                l[1] = 1;
                l[2] = 0;
            } else if ((c[1] >= 0.0 && c[0] < 0.0 && c[2] < 0.0) || (c[1] < 0.0 && c[0] >= 0.0 && c[2] >= 0.0)) {
                l[0] = 0;
                l[1] = 2;
                l[2] = 1;
            } else if ((c[2] >= 0.0 && c[0] < 0.0 && c[1] < 0.0) || (c[2] < 0.0 && c[0] >= 0.0 && c[1] >= 0.0)) {
                l[0] = 0;
                l[1] = 1;
                l[2] = 2;
            }

            p[0] = c[l[2]] / (c[l[2]] - c[l[0]]);
            p[1] = c[l[2]] / (c[l[2]] - c[l[1]]);

            if (vof) {
                *vof = p[0] * p[1];

                // Then this is actually the amount in the outer domain.
                if (c[l[2]] >= 0.0) {
                    *vof = 1.0 - *vof;
                }
            }

            if (faceLength) {
                // Physical coordinates of the crossing point
                PetscReal x1[2], x2[2];
                x1[0] = coords[l[2] * 2 + 0] + p[0] * (coords[l[0] * 2 + 0] - coords[l[2] * 2 + 0]);
                x1[1] = coords[l[2] * 2 + 1] + p[0] * (coords[l[0] * 2 + 1] - coords[l[2] * 2 + 1]);

                x2[0] = coords[l[2] * 2 + 0] + p[1] * (coords[l[1] * 2 + 0] - coords[l[2] * 2 + 0]);
                x2[1] = coords[l[2] * 2 + 1] + p[1] * (coords[l[1] * 2 + 1] - coords[l[2] * 2 + 1]);

                *faceLength = PetscSqrtReal(PetscSqr(x1[0] - x2[0]) + PetscSqr(x1[1] - x2[1]));
            }
        }
    }

    if (cellArea) *cellArea = CellArea_Triangle(coords);
}

void VOF_2D_Tri_Test() {
    const PetscReal coords[] = {0.5, 0.0, 1.0, 1.0, -2.0, 0.0};

    PetscReal vof = -1.0, area = -1.0, length = -1.0;
    const PetscReal trueArea = 5.0 / 4.0;
    const PetscInt nCases = 14;
    const PetscReal trueVof[] = {1.0, 0.0, 1.0 / 6.0, 5.0 / 6.0, 4.0 / 9.0, 5.0 / 9.0, 1.0 / 6.0, 5.0 / 6.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    const PetscReal trueLength[] = {0.0, 0.0, PetscSqrtReal(305.0) / 12.0, PetscSqrtReal(305.0) / 12.0, 5.0 / 3.0, 5.0 / 3.0, 5.0 / 12.0, 5.0 / 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const PetscReal c[] = {
        -1.0, -1.0, -1.0,  // 1
        1.0,  1.0,  1.0,   // 0
        -1.0, 2.0,  1.0,   // 1/6
        1.0,  -2.0, -1.0,  // 5/6
        1.0,  -2.0, 1.0,   // 4/9
        -1.0, 2.0,  -1.0,  // 5/9
        1.0,  2.0,  -1.0,  // 1/6
        -1.0, -2.0, 1.0,   // 5/6
        0.0,  0.0,  -1.0,  // 1
        0.0,  0.0,  1.0,   // 0
        0.0,  -1.0, 0.0,   // 1
        0.0,  1.0,  0.0,   // 0
        -1.0, 0.0,  0.0,   // 1
        1.0,  0.0,  0.0,   // 0
    };

    printf(" Starting to test 2D Tri\n");
    printf("   -------- VOF ---------   ------ Area ----------\n");
    for (PetscInt i = 0; i < nCases; ++i) {
        VOF_2D_Tri(coords, &c[i * 3], &vof, &length, &area);
        printf("%4d: %d %+.6f  %+.6f   %d %+.6f  %+.6f\n", i, PetscAbsReal(vof - trueVof[i]) < 1.e-8, vof, trueVof[i], PetscAbsReal(length - trueLength[i]) < 1.e-8, length, trueLength[i]);
    }
    printf("AREA: %d\t%+f\t%+f\n", PetscAbsReal(area - trueArea) < 1.e-8, area, trueArea);
}

// 2D Non-Simplex: DM_POLYTOPE_QUADRILATERAL
/*
     3--------2
     |        |
     |        |
     |        |
     0--------1
*/
void VOF_2D_Quad(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceLength, PetscReal *cellArea) {
    // Triangle using vertices 3-0-1
    const PetscReal x1[6] = {coords[6], coords[7], coords[0], coords[1], coords[2], coords[3]};
    const PetscReal c1[3] = {c[3], c[0], c[1]};
    PetscReal vof1, cellArea1, faceLength1;

    // Triangle using vertices 3-2-1
    const PetscReal x2[6] = {coords[6], coords[7], coords[4], coords[5], coords[2], coords[3]};
    const PetscReal c2[3] = {c[3], c[2], c[1]};
    PetscReal vof2, cellArea2, faceLength2;

    // VOF and area of each triangle.
    VOF_2D_Tri(x1, c1, &vof1, &faceLength1, &cellArea1);
    VOF_2D_Tri(x2, c2, &vof2, &faceLength2, &cellArea2);

    if (vof) *vof = (vof1 * cellArea1 + vof2 * cellArea2) / (cellArea1 + cellArea2);
    if (faceLength) *faceLength = faceLength1 + faceLength2;
    if (cellArea) *cellArea = cellArea1 + cellArea2;
}

void VOF_2D_Quad_Test() {
    const PetscReal coords[] = {0.0, 0.0, 4.0, -1.0, 3.0, 2.0, 0.5, 1.0};
    PetscReal vof = -1.0, area = -1.0, length = -1.0;
    const PetscReal trueArea = 13.0 / 2.0;
    const PetscInt nCases = 10;
    const PetscReal trueVof[] = {0.0, 1.0, 9.0 / 832.0, 823.0 / 832.0, 82.0 / 273.0, 191.0 / 273.0, 161.0 / 260.0, 99.0 / 260.0, 1822.0 / 2873.0, 1051.0 / 2873.0};
    const PetscReal trueLength[] = {0.0,
                                    0.0,
                                    9.0 / 16.0,
                                    9.0 / 16.0,
                                    38.0 * PetscSqrtReal(2.0) / 21.0,
                                    38.0 * PetscSqrtReal(2.0) / 21.0,
                                    33.0 / (10.0 * PetscSqrtReal(2.0)),
                                    33.0 / (10.0 * PetscSqrtReal(2.0)),
                                    113.0 * PetscSqrtReal(37.0) / 221.0,
                                    113.0 * PetscSqrtReal(37.0) / 221.0};
    const PetscReal c[] = {1.0, 1.0,  1.0,  1.0, -1.0, -1.0, -1.0, -1.0, -0.25, 15.0 / 4.0, 11.0 / 4.0, 0.25, 0.25, -15.0 / 4.0, -11.0 / 4.0, -0.25,      -2.0, 1.0,       3.0,  -0.5,
                           2.0, -1.0, -3.0, 0.5, -2.0, 3.0,  -1.0, -2.5, 2.0,   -3.0,       1.0,        2.5,  -1.0, -4.0 / 3.0,  1.5,         1.0 / 12.0, 1.0,  4.0 / 3.0, -1.5, -1.0 / 12.0};

    printf(" Starting to test 2D Quad\n");
    printf("   -------- VOF ---------   ------ Area ----------\n");
    for (PetscInt i = 0; i < nCases; ++i) {
        VOF_2D_Quad(coords, &c[i * 4], &vof, &length, &area);
        printf("%4d: %d %+.6f  %+.6f   %d %+.6f  %+.6f\n", i, PetscAbsReal(vof - trueVof[i]) < 1.e-8, vof, trueVof[i], PetscAbsReal(length - trueLength[i]) < 1.e-8, length, trueLength[i]);
    }
    printf("AREA: %d\t%+f\t%+f\n", PetscAbsReal(area - trueArea) < 1.e-8, area, trueArea);
}

// Move stuff like this to the solver
//  3D Simplex: DM_POLYTOPE_TETRAHEDRON
/*
         3
         ^
        /|\
       / | \
      /  |  \
     0'-.|.-'2
         1
*/
void VOF_3D_Tetra(const PetscReal coords[12], const PetscReal c[4], PetscReal *vof, PetscReal *faceArea, PetscReal *cellVol) {
    if (vof || faceArea) {
        if (c[0] >= 0.0 && c[1] >= 0.0 && c[2] >= 0.0 && c[3] >= 0.0) {
            if (vof) *vof = 0.0;
            if (faceArea) *faceArea = 0.0;

        } else if (c[0] <= 0.0 && c[1] <= 0.0 && c[2] <= 0.0 && c[3] <= 0.0) {
            if (vof) *vof = 1.0;
            if (faceArea) *faceArea = 0.0;
        } else {
            PetscBool twoNodes;
            PetscInt l[4];

            // Determine the vertex permutation so that nodes 0 (and maybe 3) are of opposite sign from nodes 1 and 2 (and possibly 3)
            if ((c[0] >= 0.0 && c[1] < 0.0 && c[2] < 0.0 && c[3] < 0.0) ||    // Case 1
                (c[0] < 0.0 && c[1] >= 0.0 && c[2] >= 0.0 && c[3] >= 0.0)) {  // Case 2
                l[0] = 1;
                l[1] = 3;
                l[2] = 2;
                l[3] = 0;
                twoNodes = PETSC_FALSE;
            } else if ((c[1] >= 0.0 && c[0] < 0.0 && c[2] < 0.0 && c[3] < 0.0) ||    // Case 3
                       (c[1] < 0.0 && c[0] >= 0.0 && c[2] >= 0.0 && c[3] >= 0.0)) {  // Case 4
                l[0] = 0;
                l[1] = 2;
                l[2] = 3;
                l[3] = 1;
                twoNodes = PETSC_FALSE;
            } else if ((c[0] >= 0.0 && c[1] >= 0.0 && c[2] < 0.0 && c[3] < 0.0) ||  // Case 5
                       (c[0] < 0.0 && c[1] < 0.0 && c[2] >= 0.0 && c[3] >= 0.0)) {  // Case 6
                l[0] = 0;
                l[1] = 2;
                l[2] = 3;
                l[3] = 1;
                twoNodes = PETSC_TRUE;
            } else if ((c[2] >= 0.0 && c[0] < 0.0 && c[1] < 0.0 && c[3] < 0.0) ||    // Case 7
                       (c[2] < 0.0 && c[0] >= 0.0 && c[1] >= 0.0 && c[3] >= 0.0)) {  // Case 8
                l[0] = 0;
                l[1] = 3;
                l[2] = 1;
                l[3] = 2;
                twoNodes = PETSC_FALSE;
            } else if ((c[0] >= 0.0 && c[2] >= 0.0 && c[1] < 0.0 && c[3] < 0.0) ||  // Case 9
                       (c[0] < 0.0 && c[2] < 0.0 && c[1] >= 0.0 && c[3] >= 0.0)) {  // Case 10
                l[0] = 0;
                l[1] = 3;
                l[2] = 1;
                l[3] = 2;
                twoNodes = PETSC_TRUE;
            } else if ((c[3] >= 0.0 && c[0] < 0.0 && c[1] < 0.0 && c[2] < 0.0) ||    // Case 11
                       (c[3] < 0.0 && c[0] >= 0.0 && c[1] >= 0.0 && c[2] >= 0.0)) {  // Case 12
                l[0] = 0;
                l[1] = 1;
                l[2] = 2;
                l[3] = 3;
                twoNodes = PETSC_FALSE;
            } else if ((c[0] >= 0.0 && c[3] >= 0.0 && c[1] < 0.0 && c[2] < 0.0) ||  // Case 13
                       (c[0] < 0.0 && c[3] < 0.0 && c[1] >= 0.0 && c[2] >= 0.0)) {  // Case 14
                l[0] = 0;
                l[1] = 1;
                l[2] = 2;
                l[3] = 3;
                twoNodes = PETSC_TRUE;
            } else {
                throw std::logic_error("Unable to determine the condition for tetrahedral volume calculation.\n");
            }

            if (twoNodes) {
                if (vof) {
                    const PetscReal d31 = c[l[3]] - c[l[1]], d32 = c[l[3]] - c[l[2]], d01 = c[l[0]] - c[l[1]], d02 = c[l[0]] - c[l[2]];
                    *vof = (d31 * d32 * c[l[0]] * c[l[0]] - d32 * c[l[0]] * c[l[1]] * c[l[3]] - d01 * c[l[2]] * c[l[3]] * c[l[3]]) / (d01 * d02 * d31 * d32);
                }

                if (faceArea) {
                    PetscReal p[4];
                    PetscReal x0[3], x1[3], x2[3], x3[3];
                    PetscReal r1[3], r2[3], r3[3];

                    p[0] = c[l[3]] / (c[l[3]] - c[l[2]]);
                    p[1] = c[l[3]] / (c[l[3]] - c[l[1]]);
                    p[2] = c[l[0]] / (c[l[0]] - c[l[2]]);
                    p[3] = c[l[0]] / (c[l[0]] - c[l[1]]);

                    // Crossing locations
                    for (PetscInt i = 0; i < 3; ++i) {
                        x0[i] = coords[l[3] * 3 + i] + p[0] * (coords[l[2] * 3 + i] - coords[l[3] * 3 + i]);
                        x1[i] = coords[l[3] * 3 + i] + p[1] * (coords[l[1] * 3 + i] - coords[l[3] * 3 + i]);

                        x2[i] = coords[l[0] * 3 + i] + p[2] * (coords[l[2] * 3 + i] - coords[l[0] * 3 + i]);
                        x3[i] = coords[l[0] * 3 + i] + p[3] * (coords[l[1] * 3 + i] - coords[l[0] * 3 + i]);
                    }

                    // Triangle-1: x1 - x0 - x2. Origin is x0
                    // Vector from the origin to one point
                    for (PetscInt i = 0; i < 3; ++i) {
                        r1[i] = x1[i] - x0[i];
                        r2[i] = x2[i] - x0[i];
                    }
                    // Cross product of the two vectors
                    r3[0] = r1[1] * r2[2] - r1[2] * r2[1];
                    r3[1] = r1[2] * r2[0] - r1[0] * r2[2];
                    r3[2] = r1[0] * r2[1] - r1[1] * r2[0];

                    *faceArea = 0.5 * PetscSqrtReal(PetscSqr(r3[0]) + PetscSqr(r3[1]) + PetscSqr(r3[2]));

                    // Triangle-2: x1 - x3 - x2. Origin is x3
                    for (PetscInt i = 0; i < 3; ++i) {
                        r1[i] = x1[i] - x3[i];
                        r2[i] = x2[i] - x3[i];
                    }
                    // Cross product of the two vectors
                    r3[0] = r1[1] * r2[2] - r1[2] * r2[1];
                    r3[1] = r1[2] * r2[0] - r1[0] * r2[2];
                    r3[2] = r1[0] * r2[1] - r1[1] * r2[0];

                    *faceArea += 0.5 * PetscSqrtReal(PetscSqr(r3[0]) + PetscSqr(r3[1]) + PetscSqr(r3[2]));
                }
            } else {
                PetscReal p[3];

                p[0] = c[l[3]] / (c[l[3]] - c[l[0]]);
                p[1] = c[l[3]] / (c[l[3]] - c[l[1]]);
                p[2] = c[l[3]] / (c[l[3]] - c[l[2]]);

                if (vof) *vof = p[0] * p[1] * p[2];

                if (faceArea) {
                    PetscReal x0, r1[3], r2[3], r3[3];
                    for (PetscInt i = 0; i < 3; ++i) {
                        // Location of the "origin"
                        x0 = coords[l[3] * 3 + i] + p[0] * (coords[l[0] * 3 + i] - coords[l[3] * 3 + i]);

                        // Vector from the origin to one point
                        r1[i] = coords[l[3] * 3 + i] + p[1] * (coords[l[1] * 3 + i] - coords[l[3] * 3 + i]) - x0;

                        // Vector from the origin to the second point
                        r2[i] = coords[l[3] * 3 + i] + p[2] * (coords[l[2] * 3 + i] - coords[l[3] * 3 + i]) - x0;
                    }

                    // Cross product of the two vectors
                    r3[0] = r1[1] * r2[2] - r1[2] * r2[1];
                    r3[1] = r1[2] * r2[0] - r1[0] * r2[2];
                    r3[2] = r1[0] * r2[1] - r1[1] * r2[0];

                    *faceArea = 0.5 * PetscSqrtReal(PetscSqr(r3[0]) + PetscSqr(r3[1]) + PetscSqr(r3[2]));
                }
            }

            if (vof && c[l[3]] >= 0.0) {
                *vof = 1.0 - *vof;
            }
        }
    }

    if (cellVol) *cellVol = CellVolume_Tetrahedron(coords);
}

void VOF_3D_Tetra_Test() {
    const PetscReal coords[] = {0.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0, 0.5, 2.0, 0.0, 1.0};

    PetscReal vof = -1.0, vol = -1.0, area = -1.0;
    const PetscReal trueVol = 5.0 / 12.0;
    const PetscInt nCases = 22;
    const PetscReal trueVof[] = {0.0,        1.0,         7.0 / 8.0,  1.0 / 8.0,   7.0 / 8.0,  1.0 / 8.0, 7.0 / 8.0, 1.0 / 8.0, 7.0 / 8.0, 1.0 / 8.0, 11.0 / 18.0,
                                 7.0 / 18.0, 11.0 / 18.0, 7.0 / 18.0, 11.0 / 18.0, 7.0 / 18.0, 0.0,       1.0,       0.0,       1.0,       0.0,       1.0};
    const PetscReal trueArea[] = {0.0,
                                  0.0,
                                  PetscSqrtReal(89.0) / 32.0,
                                  PetscSqrtReal(89.0) / 32.0,
                                  PetscSqrtReal(1.5) / 4.0,
                                  PetscSqrtReal(1.5) / 4.0,
                                  PetscSqrtReal(21.0) / 16.0,
                                  PetscSqrtReal(21.0) / 16.0,
                                  9.0 / 32.0,
                                  9.0 / 32.0,
                                  5.0 * PetscSqrtReal(683.0 / 3.0) / 96.0,
                                  5.0 * PetscSqrtReal(683.0 / 3.0) / 96.0,
                                  5.0 * PetscSqrtReal(41.0) / 96.0,
                                  5.0 * PetscSqrtReal(41.0) / 96.0,
                                  25.0 / 32.0,
                                  25.0 / 32.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0,
                                  0.0};
    const PetscReal c[] = {
        1.0,  1.0,  1.0,  1.0,   // 0
        -1.0, -1.0, -1.0, -1.0,  // 1
        1.0,  -1.0, -1.0, -1.0,  // Case 1, 7/8
        -1.0, 1.0,  1.0,  1.0,   // Case 2, 1/8
        -1.0, 1.0,  -1.0, -1.0,  // Case 3, 7/8
        1.0,  -1.0, 1.0,  1.0,   // Case 4, 1/8
        -1.0, -1.0, 1.0,  -1.0,  // Case 7, 7/8
        1.0,  1.0,  -1.0, 1.0,   // Case 8, 1/8
        -1.0, -1.0, -1.0, 1.0,   // Case 11, 7/8
        1.0,  1.0,  1.0,  -1.0,  // Case 12, 1/8
        0.5,  1.0,  -1.0, -1.0,  // Case 5, 11/18
        -0.5, -1.0, 1.0,  1.0,   // Case 6, 7/18
        0.5,  -1.0, 1.0,  -1.0,  // Case 9, 11/18
        -0.5, 1.0,  -1.0, 1.0,   // Case 10, 7/18
        0.5,  -1.0, -1.0, 1.0,   // Case 13, 11/18
        -0.5, 1.0,  1.0,  -1.0,  // Case 14, 7/18
        0.0,  0.0,  1.0,  1.0,   // 0
        0.0,  0.0,  -1.0, -1.0,  // 1
        0.0,  1.0,  0.0,  1.0,   // 0
        0.0,  -1.0, 0.0,  -1.0,  // 1
        0.0,  1.0,  1.0,  0.0,   // 0
        0.0,  -1.0, -1.0, 0.0    // 1
    };

    printf(" Starting to test 3D Tetra\n");
    printf("      -------- VOF ---------   ------ Area ----------\n");
    for (PetscInt i = 0; i < nCases; ++i) {
        VOF_3D_Tetra(coords, &c[i * 4], &vof, &area, &vol);
        printf("%4d: %d %+.6f  %+.6f   %d %+.6f  %+.6f\n", i, PetscAbsReal(vof - trueVof[i]) < 1.e-8, vof, trueVof[i], PetscAbsReal(area - trueArea[i]) < 1.e-8, area, trueArea[i]);
    }
    printf("VOL: %d\t%+f\t%+f\n", PetscAbsReal(vol - trueVol) < 1.e-8, vol, trueVol);
}

// 3D Non-Simplex: DM_POLYTOPE_HEXAHEDRON
/*
        7--------6
       /|       /|
      / |      / |
     4--------5  |
     |  |     |  |
     |  |     |  |
     |  1--------2
     | /      | /
     |/       |/
     0--------3
*/
// Uses "HOW TO SUBDIVIDE PYRAMIDS, PRISMS AND HEXAHEDRA INTO TETRAHEDRA" to get the divisions.
// Note that the numbering differs between the paper and DMPLEX:
// V1->0, V2->3, V3->2, V4->1, V5->4, V6->5, V7->6, V8->7
void VOF_3D_Hex(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *faceArea, PetscReal *cellVol) {
    const PetscInt nTet = 5, nVerts = 4, dim = 3;
    PetscInt t, v, d, vid;
    PetscReal x[nVerts * dim], f[nVerts], tetArea, tetVOF, tetVOL, sumVOF = 0.0, sumVOL = 0.0, sumArea = 0.0;
    PetscInt TID[nTet * nVerts] = {0, 3, 2, 5,   // Tet1
                                   2, 7, 5, 6,   // Tet2
                                   0, 4, 7, 5,   // Tet3
                                   0, 1, 2, 7,   // Tet4
                                   0, 2, 7, 5};  // Tet5

    for (t = 0; t < nTet; ++t) {         // Iterate over all tetrahedrons
        for (v = 0; v < nVerts; ++v) {   // The 4 verties
            vid = TID[t * nVerts + v];   // Hex vertex ID
            f[v] = c[vid];               // Level-set val
            for (d = 0; d < dim; ++d) {  // Set coordinates
                x[v * dim + d] = coords[vid * dim + d];
            }
        }

        VOF_3D_Tetra(x, f, &tetVOF, &tetArea, &tetVOL);
        sumVOF += tetVOF * tetVOL;
        sumArea += tetArea;
        sumVOL += tetVOL;
        //    printf("%d: %+f\t%+f\n", t, tetVOF, tetVOL);
        //    exit(0);
    }

    if (vof) *vof = sumVOF / sumVOL;
    if (faceArea) *faceArea = sumArea;
    if (cellVol) *cellVol = sumVOL;
}

void VOF_3D_Hex_Test() {
    const PetscReal coords[] = {1.0, 1.0, 1.0, 1.0, 3.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 3.0};

    PetscReal vof = -1.0, vol = -1.0, area = -1.0;
    const PetscReal trueVol = 3.0;
    const PetscInt nCases = 6;
    const PetscReal trueVof[] = {3.0 / 4.0, 1.0 / 4.0, 1.0 / 8.0, 7.0 / 8.0, 9.0 / 512.0, 503.0 / 512.0};
    const PetscReal trueArea[] = {3.0 / PetscSqrtReal(2.0), 3.0 / PetscSqrtReal(2.0), 1.5, 1.5, 27.0 * PetscSqrtReal(5.0) / 128.0, 27.0 * PetscSqrtReal(5.0) / 128.0};
    const PetscReal c[] = {
        0.0,        -2.0,  -2.0,  0.0,        1.0,         1.0,         -1.0,       -1.0,        // x-y
        0.0,        2.0,   2.0,   0.0,        -1.0,        -1.0,        1.0,        1.0,         //-x+y
        -0.5,       -0.5,  0.5,   0.5,        0.5,         1.5,         1.5,        0.5,         // x-1.5
        0.5,        0.5,   -0.5,  -0.5,       -0.5,        -1.5,        -1.5,       -0.5,        // -x+1.5
        5.0 / 4.0,  -0.75, 0.25,  9.0 / 4.0,  13.0 / 4.0,  17.0 / 4.0,  9.0 / 4.0,  5.0 / 4.0,   // 2x-y+0.25
        -5.0 / 4.0, 0.75,  -0.25, -9.0 / 4.0, -13.0 / 4.0, -17.0 / 4.0, -9.0 / 4.0, -5.0 / 4.0,  // 2x-y+0.25
    };

    printf(" Starting to test 3D Hex\n");
    printf("   -------- VOF ---------   ------ Area ----------\n");
    for (PetscInt i = 0; i < nCases; ++i) {
        VOF_3D_Hex(coords, &c[i * 8], &vof, &area, &vol);
        printf("%4d: %d %+.6f  %+.6f   %d %+.6f  %+.6f\n", i, PetscAbsReal(vof - trueVof[i]) < 1.e-8, vof, trueVof[i], PetscAbsReal(area - trueArea[i]) < 1.e-8, area, trueArea[i]);
    }
    printf("VOL: %d\t%+f\t%+f\n", PetscAbsReal(vol - trueVol) < 1.e-8, vol, trueVol);
}
