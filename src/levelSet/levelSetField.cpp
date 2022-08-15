#include "levelSetField.hpp"


using namespace ablate::levelSet;

void LevelSetField::Normal2D(PetscInt c, PetscScalar *n) {

  PetscReal             cx, cy, g;
  Vec                   phi = LevelSetField::phi;
  std::shared_ptr<RBF>  rbf = LevelSetField::rbf;

  cx = rbf->EvalDer(phi, c, 1, 0, 0);
  cy = rbf->EvalDer(phi, c, 0, 1, 0);
  g = PetscSqrtReal(cx*cx + cy*cy);

  n[0] = cx/g;
  n[1] = cy/g;


}

void LevelSetField::Normal3D(PetscInt c, PetscReal *n) {

  PetscReal             cx, cy, cz, g;
  Vec                   phi = LevelSetField::phi;
  std::shared_ptr<RBF>  rbf = LevelSetField::rbf;

  cx = rbf->EvalDer(phi, c, 1, 0, 0);
  cy = rbf->EvalDer(phi, c, 0, 1, 0);
  cz = rbf->EvalDer(phi, c, 0, 0, 1);
  g = sqrt(cx*cx + cy*cy + cz*cz);

  n[0] = cx/g;
  n[1] = cy/g;
  n[2] = cz/g;
}

PetscReal LevelSetField::Curvature2D(PetscInt c) {

  PetscReal             k = 0.0;
  PetscReal             cx, cy, cxx, cyy, cxy;
  Vec                   phi = LevelSetField::phi;
  std::shared_ptr<RBF>  rbf = LevelSetField::rbf;

  cx = rbf->EvalDer(phi, c, 1, 0, 0);
  cy = rbf->EvalDer(phi, c, 0, 1, 0);
  cxx = rbf->EvalDer(phi, c, 2, 0, 0);
  cyy = rbf->EvalDer(phi, c, 0, 2, 0);
  cxy = rbf->EvalDer(phi, c, 1, 1, 0);

  k = (cxx*cy*cy + cyy*cx*cx - 2.0*cxy*cx*cy)/pow(cx*cx+cy*cy,1.5);

  return k;
}

PetscReal LevelSetField::Curvature3D(PetscInt c) {

  PetscReal             k = 0.0;
  PetscReal             cx, cy, cz;
  PetscReal             cxx, cyy, czz;
  PetscReal             cxy, cxz, cyz;
  Vec                   phi = LevelSetField::phi;
  std::shared_ptr<RBF>  rbf = LevelSetField::rbf;

  cx = rbf->EvalDer(phi, c, 1, 0, 0);
  cy = rbf->EvalDer(phi, c, 0, 1, 0);
  cz = rbf->EvalDer(phi, c, 0, 0, 1);
  cxx = rbf->EvalDer(phi, c, 2, 0, 0);
  cyy = rbf->EvalDer(phi, c, 0, 2, 0);
  czz = rbf->EvalDer(phi, c, 0, 0, 2);
  cxy = rbf->EvalDer(phi, c, 1, 1, 0);
  cxz = rbf->EvalDer(phi, c, 1, 0, 1);
  cyz = rbf->EvalDer(phi, c, 0, 1, 1);

  k = (cxx*(cy*cy + cz*cz) + cyy*(cx*cx + cz*cz) + czz*(cx*cx + cy*cy) - 2.0*(cxy*cx*cy + cxz*cx*cz + cyz*cy*cz))/pow(cx*cx+cy*cy+cz*cz,1.5);

  return k;
}

// There has to be a better way of doing this so that the curvature/normal function points directly to either 2D or 3D during setup.
PetscReal LevelSetField::Curvature(PetscInt c) {
  if (LevelSetField::dim==2) {
    return LevelSetField::Curvature2D(c);
  }
  else {
    return LevelSetField::Curvature3D(c);
  }
}

void LevelSetField::Normal(PetscInt c, PetscReal *n) {
  if (LevelSetField::dim==2) {
    return LevelSetField::Normal2D(c, n);
  }
  else {
    return LevelSetField::Normal3D(c, n);
  }
}


LevelSetField::LevelSetField(std::shared_ptr<domain::Region> region) : region(region) {}




LevelSetField::LevelSetField(std::shared_ptr<RBF> rbf, LevelSetField::levelSetShape shape) {

  LevelSetField::rbf = rbf;
  LevelSetField::dm = rbf->GetDM();

  PetscInt            d, dim;
  PetscInt            cStart, cEnd, c;
  PetscInt            nSet = 3;
  PetscScalar         *val;
  PetscReal           lo[] = {0.0, 0.0, 0.0}, hi[] = {0.0, 0.0, 0.0}, centroid[] = {0.0, 0.0, 0.0}, pos[] = {0.0, 0.0, 0.0};
  PetscReal           radius = 1.0;



  DMGetDimension(dm, &dim) >> ablate::checkError;
  LevelSetField::dim = dim;




  // Create the vectors
  DMCreateGlobalVector(dm, &(LevelSetField::phi)) >> ablate::checkError;
  DMCreateGlobalVector(dm, &(LevelSetField::curv)) >> ablate::checkError;

  PetscInt lsz, gsz;
  VecGetLocalSize(phi, &lsz) >> ablate::checkError;
  VecGetSize(phi, &gsz) >> ablate::checkError;
  VecCreateMPI(PETSC_COMM_WORLD, dim*lsz, dim*gsz, &(LevelSetField::normal)) >> ablate::checkError;
  VecSetBlockSize(LevelSetField::normal, dim) >> ablate::checkError;

  DMGetBoundingBox(dm, lo, hi) >> ablate::checkError;
  for (d = 0; d < dim; ++d) {
    centroid[d] = 0.5*(lo[d] + hi[d]);
  }

  PetscOptionsGetReal(NULL, NULL, "-radius", &(radius), NULL) >> ablate::checkError;
  PetscOptionsGetRealArray(NULL, NULL, "-centroid", centroid, &nSet, NULL) >> ablate::checkError;


  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> ablate::checkError;       // Range of cells
  VecGetArray(LevelSetField::phi, &val) >> ablate::checkError;
  // Set the initial shape.
  switch (shape) {
    case LevelSetField::levelSetShape::SPHERE:
     for (c = cStart; c < cEnd; ++c) {
       DMPlexComputeCellGeometryFVM(dm, c, NULL, pos, NULL) >> ablate::checkError;
       val[c - cStart] = LevelSetField::Sphere(pos, centroid, radius);
      }

      break;
    case LevelSetField::levelSetShape::ELLIPSE:
      for (c = cStart; c < cEnd; ++c) {
        DMPlexComputeCellGeometryFVM(dm, c, NULL, pos, NULL) >> ablate::checkError;
        val[c - cStart] = LevelSetField::Ellipse(pos, centroid, radius);
      }

      break;
    case LevelSetField::levelSetShape::STAR:
      for (c = cStart; c < cEnd; ++c) {
        DMPlexComputeCellGeometryFVM(dm, c, NULL, pos, NULL) >> ablate::checkError;
        val[c - cStart] = LevelSetField::Star(pos, centroid);
      }

      break;
    default:
      throw std::invalid_argument("Unknown level set shape shape");
  }

  VecRestoreArray(LevelSetField::phi, &val) >> ablate::checkError;

  VecGhostUpdateBegin(LevelSetField::phi, INSERT_VALUES, SCATTER_FORWARD) >> ablate::checkError;
  VecGhostUpdateEnd(LevelSetField::phi, INSERT_VALUES, SCATTER_FORWARD) >> ablate::checkError;

  // Now setup the derivatives and set the curvature/normal calculations
  PetscInt nDer = 0;
  PetscInt dx[10], dy[10], dz[10];

  nDer = ( dim == 2 ) ? 5 : 10;
  PetscInt i = 0;
  dx[i] = 1; dy[i] = 0; dz[i++] = 0;
  dx[i] = 0; dy[i] = 1; dz[i++] = 0;
  dx[i] = 2; dy[i] = 0; dz[i++] = 0;
  dx[i] = 0; dy[i] = 2; dz[i++] = 0;
  dx[i] = 1; dy[i] = 1; dz[i++] = 0;
  if( dim == 3) {
    dx[i] = 0; dy[i] = 0; dz[i++] = 1;
    dx[i] = 0; dy[i] = 0; dz[i++] = 2;
    dx[i] = 1; dy[i] = 0; dz[i++] = 1;
    dx[i] = 0; dy[i] = 1; dz[i++] = 1;
    dx[i] = 1; dy[i] = 1; dz[i++] = 1;
  }
  rbf->SetDerivatives(nDer, dx, dy, dz);

}

LevelSetField::~LevelSetField() {
  VecDestroy(&(LevelSetField::phi));
  VecDestroy(&(LevelSetField::normal));
  VecDestroy(&(LevelSetField::curv));


}

void LevelSetField::ComputeAllNormal() {
  PetscScalar *val;
  PetscInt    cStart, cEnd, c;

  DMPlexGetHeightStratum(LevelSetField::dm, 0, &cStart, &cEnd) >> ablate::checkError;       // Range of cells

  VecGetArray(LevelSetField::normal, &val) >> ablate::checkError;
  for (c = cStart; c < cEnd; ++c) {
    LevelSetField::Normal(c, &val[(c - cStart)*dim]);
  }
  VecRestoreArray(LevelSetField::normal, &val) >> ablate::checkError;
//  VecGhostUpdateBegin(LevelSetField::normal, INSERT_VALUES, SCATTER_FORWARD) >> ablate::checkError;
//  VecGhostUpdateEnd(LevelSetField::normal, INSERT_VALUES, SCATTER_FORWARD) >> ablate::checkError;

}

void LevelSetField::ComputeAllCurvature() {
  PetscScalar *val;
  PetscInt    cStart, cEnd, c;

  DMPlexGetHeightStratum(LevelSetField::dm, 0, &cStart, &cEnd) >> ablate::checkError;       // Range of cells

  VecGetArray(LevelSetField::curv, &val) >> ablate::checkError;
  for (c = cStart; c < cEnd; ++c) {
    val[c - cStart] = LevelSetField::Curvature(c);
  }
  VecRestoreArray(LevelSetField::curv, &val) >> ablate::checkError;
  VecGhostUpdateBegin(LevelSetField::curv, INSERT_VALUES, SCATTER_FORWARD) >> ablate::checkError;
  VecGhostUpdateEnd(LevelSetField::curv, INSERT_VALUES, SCATTER_FORWARD) >> ablate::checkError;
}

std::vector<std::shared_ptr<ablate::domain::FieldDescription>> LevelSetField::GetFields() {
    std::vector<std::shared_ptr<ablate::domain::FieldDescription>> levelSetField{
        std::make_shared<domain::FieldDescription>("level set field", "phi", domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::SOL, domain::FieldType::FVM, region)};

  return levelSetField;
}

PetscReal LevelSetField::Interpolate(PetscScalar xyz[3]) {
  std::shared_ptr<RBF>  rbf = LevelSetField::rbf;
  DMInterpolationInfo   ctx;
  DM                    dm = rbf->GetDM();
  PetscInt              c = -1;
  Vec                   phi = LevelSetField::phi;
  PetscReal             val;

  DMInterpolationCreate(PETSC_COMM_WORLD, &ctx) >> ablate::checkError;
  DMInterpolationSetDim(ctx, LevelSetField::dim) >> ablate::checkError;
  DMInterpolationAddPoints(ctx, 1, xyz) >> ablate::checkError;
  DMInterpolationSetUp(ctx, dm, PETSC_FALSE, PETSC_TRUE) >> ablate::checkError;
  c = ctx->cells[0];
  DMInterpolationDestroy(&ctx) >> ablate::checkError;

  val = rbf->Interpolate(phi, c, xyz);

  return val;
}

PetscReal LevelSetField::Interpolate(const PetscReal x, const double y, const double z) {

  PetscReal xyz[3] = {x, y, z};
  PetscReal val = LevelSetField::Interpolate(xyz);

  return val;
}



void LevelSetField::Advect(Vec velocity, const PetscReal dt) {

  Vec               phi = LevelSetField::phi, nextPhi = nullptr;
  DM                dm = LevelSetField::dm;
  PetscInt          dim = LevelSetField::dim;
  PetscInt          cStart, cEnd, c, cShift;
  PetscScalar       *newVal;
  const PetscScalar *vel;
  PetscReal         pos[3] = {0.0, 0.0, 0.0};


  VecDuplicate(phi, &nextPhi);

  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd) >> ablate::checkError;       // Range of cells

  VecGetArray(nextPhi, &newVal) >> ablate::checkError;
  VecGetArrayRead(velocity, &vel) >> ablate::checkError;
  for (c = cStart; c < cEnd; ++c) {
    cShift = c - cStart;
    // Cell center
    DMPlexComputeCellGeometryFVM(dm, c, NULL, pos, NULL) >> ablate::checkError;

    // Step backward
    for (PetscInt d = 0; d < dim; ++d) {
      pos[d] -= dt*vel[cShift*dim + d];
    }

    newVal[cShift] = LevelSetField::Interpolate(pos);
  }
  VecRestoreArrayRead(velocity, &vel) >> ablate::checkError;
  VecRestoreArray(nextPhi, &newVal) >> ablate::checkError;

  VecCopy(nextPhi, phi) >> ablate::checkError;
  VecDestroy(&nextPhi) >> ablate::checkError;

  VecGhostUpdateBegin(phi, INSERT_VALUES, SCATTER_FORWARD) >> ablate::checkError;
  VecGhostUpdateEnd(phi, INSERT_VALUES, SCATTER_FORWARD) >> ablate::checkError;


  VecDestroy(&nextPhi);



}


// Area of a triangle.
// It's 1/2 of the determinant of
//   | x1 x2 |
//   | y1 y2 |
// where the coordinates are shifted so that x[0] is at the origin.
PetscReal CellArea_Triangle(const PetscReal coords[]) {
  const PetscReal x1 = coords[2] - coords[0], y1 = coords[3] - coords[1];
  const PetscReal x2 = coords[4] - coords[0], y2 = coords[5] - coords[1];
  const PetscReal area = 0.5*(x1*y2 - y1*x2);

  return PetscAbsReal(area);
}

// Volume of a tetrahedron.
// It's 1/6 of the determinant of
//   | x1 x2 x3 |
//   | y1 y2 y3 |
//   | z1 z2 z3 |
// where the coordinates are shifted so that x[0] is at the origin.
PetscReal CellVolume_Tetrahedron(const PetscReal coords[]) {
  const PetscReal x1 = coords[3] - coords[0], y1 = coords[4]  - coords[1], z1 = coords[5]  - coords[2];
  const PetscReal x2 = coords[6] - coords[0], y2 = coords[7]  - coords[1], z2 = coords[8]  - coords[2];
  const PetscReal x3 = coords[9] - coords[0], y3 = coords[10] - coords[1], z3 = coords[11] - coords[2];
  const PetscReal vol = (x2*y3*z1 + x3*y1*z2 + x1*y2*z3 - x3*y2*z1 - x2*y1*z3 - x1*y3*z2)/6.0;

  return PetscAbsReal(vol);
}


// 2D Simplex: DM_POLYTOPE_TRIANGLE
// Note: The article returns the area of the unit triangle. To get the actual volume you would
//    multiply the value by 2*(volume of the cell). Since we're interested in the VOF we simply multiply
//    the value by 2, as we would be dividing by the volume of the cell to get the VOF.
void VOF_2D_Tri(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *cellArea) {


  if (vof) {
    PetscReal p[3];
    if (c[0] < 0.0 && c[1] < 0.0 && c[2] < 0.0) {
      *vof = 1.0;
    } else if (c[0] >= 0.0 && c[1] >= 0.0 && c[2] >= 0.0) {
      *vof = 0.0;
    } else {
      if ((c[0] >= 0.0 && c[1] < 0.0 && c[2] < 0.0) || (c[0] < 0.0 && c[1] >= 0.0 && c[2] >= 0.0)) {
        p[0] = c[2];
        p[1] = c[1];
        p[2] = c[0];
      } else if ((c[1] >= 0.0 && c[0]<0.0 && c[2]<0.0) || (c[1] < 0.0 && c[0] >= 0.0 && c[2] >= 0.0)) {
        p[0] = c[0];
        p[1] = c[2];
        p[2] = c[1];
      } else if ((c[2] >= 0.0 && c[0] < 0.0 && c[1] < 0.0) || (c[2] < 0.0 && c[0] >= 0.0 && c[1] >= 0.0)) {
        p[0] = c[0];
        p[1] = c[1];
        p[2] = c[2];
      }

      *vof = p[2]*p[2]/((p[2] - p[0])*(p[2] - p[1]));

      // Then this is actually the amount in the outer domain.
      if (p[2] >= 0.0) {
        *vof = 1.0 - *vof;
      }
    }
  }

  if (cellArea) *cellArea = CellArea_Triangle(coords);

}

// 2D Non-Simplex: DM_POLYTOPE_QUADRILATERAL
/*
     3--------2
     |        |
     |        |
     |        |
     0--------1
*/
void VOF_2D_Quad(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *cellArea) {
  // Triangle using vertices 0-1-3
  const PetscReal x1[6] = {coords[0], coords[1], coords[2], coords[3], coords[6], coords[7]};
  const PetscReal c1[3] = {c[0], c[1], c[3]};
  PetscReal vof1, cellArea1;

  // Triangle using vertices 2-1-3
  const PetscReal x2[6] = {coords[4], coords[5], coords[2], coords[3], coords[6], coords[7]};
  const PetscReal c2[3] = {c[2], c[1], c[3]};
  PetscReal vof2, cellArea2;

  // VOF and area of each triangle.
  VOF_2D_Tri(x1, c1, &vof1, &cellArea1);
  VOF_2D_Tri(x2, c2, &vof2, &cellArea2);

  if(vof)      *vof = (vof1*cellArea1 + vof2*cellArea2) / (cellArea1 + cellArea2);
  if(cellArea) *cellArea = cellArea1 + cellArea2;

}

// 3D Simplex: DM_POLYTOPE_TETRAHEDRON
/*
         3
         ^
        /|\
       / | \
      /  |  \
     0'-.|.-'2
         1
*/
void VOF_3D_Tetra(const PetscReal coords[12], const PetscReal c[4], PetscReal *vof, PetscReal *cellVol) {

  if (vof) {
    if ( c[0] == 0.0 && c[1] == 0.0 && c[2] == 0.0 && c[3] == 0.0) {
        throw std::logic_error("It appears that all nodes of the tetrahedron have a zero level set value!\n");
    } else if ( c[0] <= 0.0 && c[1] <= 0.0 && c[2] <= 0.0 && c[3] <= 0.0) {
      *vof = 1.0;
    } else if ( c[0] >= 0.0 && c[1] >= 0.0 && c[2] >= 0.0 && c[3] >= 0.0) {
      *vof = 0.0;
    } else {
      PetscReal p[4];
      PetscBool twoNodes;

      if ( (c[0] >= 0.0 && c[1] <  0.0 && c[2] <  0.0 && c[3] <  0.0) ||  // Case 1
           (c[0] <  0.0 && c[1] >= 0.0 && c[2] >= 0.0 && c[3] >= 0.0) ) { // Case 2
        p[3] = c[0];
        p[0] = c[1];
        p[2] = c[2];
        p[1] = c[3];
        twoNodes = PETSC_FALSE;
      } else if ( (c[1] >= 0.0 && c[0] <  0.0 && c[2] <  0.0 && c[3] <  0.0) ||  // Case 3
                  (c[1] <  0.0 && c[0] >= 0.0 && c[2] >= 0.0 && c[3] >= 0.0) ) { // Case 4
        p[0] = c[0];
        p[3] = c[1];
        p[1] = c[2];
        p[2] = c[3];
        twoNodes = PETSC_FALSE;
      } else if ( (c[0] >= 0.0 && c[1] >= 0.0 && c[2] <  0.0 && c[3] <  0.0) ||  // Case 5
                  (c[0] <  0.0 && c[1] <  0.0 && c[2] >= 0.0 && c[3] >= 0.0) ) { // Case 6
        p[0] = c[0];
        p[3] = c[1];
        p[1] = c[2];
        p[2] = c[3];
        twoNodes = PETSC_TRUE;
      } else if ( (c[2] >= 0.0 && c[0] <  0.0 && c[1] <  0.0 && c[3] <  0.0) ||  // Case 7
                  (c[2] <  0.0 && c[0] >= 0.0 && c[1] >= 0.0 && c[3] >= 0.0) ) { // Case 8
        p[0] = c[0];
        p[2] = c[1];
        p[3] = c[2];
        p[1] = c[3];
        twoNodes = PETSC_FALSE;
      } else if ( (c[0] >= 0.0 && c[2] >= 0.0 && c[1] <  0.0 && c[3] <  0.0) ||  // Case 9
                  (c[0] <  0.0 && c[2] <  0.0 && c[1] >= 0.0 && c[3] >= 0.0) ) { // Case 10
        p[0] = c[0];
        p[2] = c[1];
        p[3] = c[2];
        p[1] = c[3];
        twoNodes = PETSC_TRUE;
      } else if ( (c[3] >= 0.0 && c[0] <  0.0 && c[1] <  0.0 && c[2] <  0.0) ||  // Case 11
                  (c[3] <  0.0 && c[0] >= 0.0 && c[1] >= 0.0 && c[2] >= 0.0) ) { // Case 12
        p[0] = c[0];
        p[1] = c[1];
        p[2] = c[2];
        p[3] = c[3];
        twoNodes = PETSC_FALSE;
      } else if ( (c[0] >= 0.0 && c[3] >= 0.0 && c[1] <  0.0 && c[2] <  0.0) ||  // Case 13
                  (c[0] <  0.0 && c[3] <  0.0 && c[1] >= 0.0 && c[2] >= 0.0) ) { // Case 14
        p[0] = c[0];
        p[1] = c[1];
        p[2] = c[2];
        p[3] = c[3];
        twoNodes = PETSC_TRUE;
      }
      else {
        throw std::logic_error("Unable to determine the condition for tetrahedral volume calculation.\n");
      }

      if (twoNodes){
        const PetscReal d31 = p[3] - p[1], d32 = p[3] - p[2], d01 = p[0] - p[1], d02 = p[0] - p[2];
        *vof = (d31*d32*p[0]*p[0] - d32*p[0]*p[1]*p[3] - d01*p[2]*p[3]*p[3])/(d01*d02*d31*d32);
      }
      else {
        *vof = p[3]*p[3]*p[3]/((p[3] - p[0])*(p[3] - p[1])*(p[3] - p[2]));
      }

      if ( p[3] >= 0.0 ) {
        *vof = 1.0 - *vof;
      }
    }
  }

  if (cellVol) *cellVol = CellVolume_Tetrahedron(coords);
}


// Test using x-y+z
void VOF_3D_Tetra_Test( ){
  const PetscReal coords[] = { 0.0, 0.0, 0.0,
                                2.0, 0.5, 0.0,
                                0.0, 1.0, 0.5,
                                2.0, 0.0, 1.0};

  PetscReal       vof = -1.0, vol = -1.0;
  const PetscReal trueVol = 5.0/12.0;
  const PetscInt  nCases = 22;
  const PetscReal trueVof[] = {0.0, 1.0, 7.0/8.0, 1.0/8.0, 7.0/8.0, 1.0/8.0, 7.0/8.0, 1.0/8.0, 7.0/8.0, 1.0/8.0, 11.0/18.0, 7.0/18.0, 11.0/18.0, 7.0/18.0, 11.0/18.0, 7.0/18.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
  const PetscReal c[] = {  1.0,  1.0,  1.0,  1.0,  // 0
                          -1.0, -1.0, -1.0, -1.0,  // 1
                           1.0, -1.0, -1.0, -1.0,  // Case 1, 1/8
                          -1.0,  1.0,  1.0,  1.0,  // Case 2, 7/8
                          -1.0,  1.0, -1.0, -1.0,  // Case 3, 1/8
                           1.0, -1.0,  1.0,  1.0,  // Case 4, 7/8
                          -1.0, -1.0,  1.0, -1.0,  // Case 7, 1/8
                           1.0,  1.0, -1.0,  1.0,  // Case 8, 7/8
                          -1.0, -1.0, -1.0,  1.0,  // Case 11, 1/8
                           1.0,  1.0,  1.0, -1.0,  // Case 12, 7/8
                           0.5,  1.0, -1.0, -1.0,  // Case 5, 7/18
                          -0.5, -1.0,  1.0,  1.0,  // Case 6, 11/18
                           0.5, -1.0,  1.0, -1.0,  // Case 9, 7/18
                          -0.5,  1.0, -1.0,  1.0,  // Case 10, 11/18
                           0.5, -1.0, -1.0,  1.0,  // Case 13, 7/18
                          -0.5,  1.0,  1.0, -1.0,  // Case 14, 11/18
                           0.0,  0.0,  1.0,  1.0,  // 0
                           0.0,  0.0, -1.0, -1.0,  // 1
                           0.0,  1.0,  0.0,  1.0,  // 0
                           0.0, -1.0,  0.0, -1.0,  // 1
                           0.0,  1.0,  1.0,  0.0,  // 0
                           0.0, -1.0, -1.0,  0.0   // 1
                        };

//  VOF_3D_Tetra(coords, &c[3*4], &vof, &vol);
//  printf("%+f\t%+f\n", vof, vol);

  for (PetscInt i = 0; i < nCases; ++i ) {
    VOF_3D_Tetra(coords, &c[i*4], &vof, &vol);
    printf("VOF: %d\t%+f\t%+f\n", PetscAbsReal(vof - trueVof[i])<1.e-8 , vof, trueVof[i]);
  }
  printf("VOL: %d\t%+f\t%+f\n", PetscAbsReal(vol - trueVol)<1.e-8 , vol, trueVol);

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
void VOF_3D_Hex(const PetscReal coords[], const PetscReal c[], PetscReal *vof, PetscReal *cellVol) {
  const PetscInt  nTet = 5, nVerts = 4, dim = 3;
  PetscInt        t, v, d, vid;
  PetscReal       x[nVerts*dim], f[nVerts], tetVOF, tetVOL, sumVOF = 0.0, sumVOL = 0.0;
//  PetscInt        TID[nTet*nVerts] = { 0, 3, 2, 5,  // Tet1
//                                       0, 2, 7, 5,  // Tet2
//                                       0, 2, 1, 7,  // Tet3
//                                       0, 5, 7, 4,  // Tet4
//                                       2, 7, 5, 6}; // Tet5

PetscInt        TID[nTet*nVerts] = { 0, 3, 2, 5,  // Tet1
                                     2, 7, 5, 6,  // Tet2
                                     0, 4, 7, 5,  // Tet3
                                     0, 1, 2, 7,  // Tet4
                                     0, 2, 7, 5}; // Tet5



  for (t = 0; t < nTet; ++t) {      // Iterate over all tetrahedrons
    for (v = 0; v < nVerts; ++v) {  // The 4 verties
      vid = TID[t*nVerts + v];      // Hex vertex ID
      f[v] = c[vid];                // Level-set val
      for (d = 0; d < dim; ++d) {   // Set coordinates
        x[v*dim + d] = coords[vid*dim + d];
      }
    }

    VOF_3D_Tetra(x, f, &tetVOF, &tetVOL);
    sumVOF += tetVOF*tetVOL;
    sumVOL += tetVOL;
//    printf("%d: %+f\t%+f\n", t, tetVOF, tetVOL);
//    exit(0);
  }

  if(vof)     *vof = sumVOF/sumVOL;
  if(cellVol) *cellVol = sumVOL;
}


void VOF_3D_Hex_Test( ){
  const PetscReal coords[] = {1.0, 1.0, 1.0,
                              1.0, 3.0, 2.0,
                              2.0, 4.0, 2.0,
                              2.0, 2.0, 1.0,
                              2.0, 1.0, 2.0,
                              3.0, 2.0, 2.0,
                              3.0, 4.0, 3.0,
                              2.0, 3.0, 3.0};

  PetscReal       vof = -1.0, vol = -1.0;
  const PetscReal trueVol = 3.0;
  const PetscInt  nCases = 4;
  const PetscReal trueVof[] = {1.0/24.0, 23.0/24.0, 0.25, 0.75};
  const PetscReal c[] = { -1.0,  1.0,  3.0,  1.0,  0.0,  2.0,  4.0,  2.0, // 1/24
                           1.0, -1.0, -3.0, -1.0,  0.0, -2.0, -4.0, -2.0, // 23/24
                          -1.0,  1.0,  2.0,  0.0, -1.0,  0.0,  2.0,  1.0, // 1/4
                           1.0, -1.0, -2.0,  0.0,  1.0,  0.0, -2.0, -1.0, // 3/4
                        };

  for (PetscInt i = 0; i < nCases; ++i ) {
    VOF_3D_Hex(coords, &c[i*8], &vof, &vol);
    printf("VOF: %d\t%+f\t%+f\n", PetscAbsReal(vof - trueVof[i])<1.e-8 , vof, trueVof[i]);
  }
  printf("VOL: %d\t%+f\t%+f\n", PetscAbsReal(vol - trueVol)<1.e-8 , vol, trueVol);

}

// Returns the VOF for a given cell. Refer to "Quadrature rules for triangular and tetrahedral elements with generalized functions"
//  by Holdych, Noble, and Secor, Int. J. Numer. Meth. Engng 2008; 73:1310-1327.
PetscReal LevelSetField::VOF(const PetscInt p) {

  DMPolytopeType    ct;
  PetscInt          dim = LevelSetField::dim, Nc, cStart, nVerts, i, j;
  PetscReal         x0[3] = {0.0, 0.0, 0.0}, n[3] = {0.0, 0.0, 0.0}, g;
  PetscReal         *c = NULL, *coords = NULL, c0, vof = -1.0;
  DM                dm = LevelSetField::dm;
  const PetscScalar *array;
  PetscBool         isDG;

  // Usually cStart is 0, but get it just in case it changes
  DMPlexGetHeightStratum(dm, 0, &cStart, NULL) >> ablate::checkError;

  // The cell center
  DMPlexComputeCellGeometryFVM(dm, p, NULL, x0, NULL) >> ablate::checkError;

  // Level-set value at cell-center
  VecGetArrayRead(LevelSetField::phi, &array) >> ablate::checkError;
  c0 = array[p - cStart];
  VecRestoreArrayRead(LevelSetField::phi, &array) >> ablate::checkError;

  // Normal vector
  VecGetArrayRead(LevelSetField::normal, &array) >> ablate::checkError;
  g = 0.0;
  for (i = 0; i < dim; ++i) {
    n[i] = array[(p - cStart)*dim + i];
    g += PetscSqr(n[i]);
  }
  VecRestoreArrayRead(LevelSetField::normal, &array) >> ablate::checkError;
  g = PetscSqrtReal(g);
  for (i = 0; i < dim; ++i) n[i] /= g;


  // Coordinates of the cell vertices
  DMPlexGetCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::checkError;

  // Number of vertices
  nVerts = Nc/dim;

  PetscMalloc(nVerts, &c) >> ablate::checkError;



  // The level set value of each vertex. This assumes that the interface is a line/plane
  //    with the given unit normal.
  for (i = 0; i < nVerts; ++i) {
    c[i] = c0;
    for (j = 0; j < dim; ++j) {
      c[i] += n[j]*(coords[i*dim + j] - x0[j]);
    }
  }

  DMPlexRestoreCellCoordinates(dm, p, &isDG, &Nc, &array, &coords) >> ablate::checkError;


  // Get the cell type and call appropriate VOF function
  DMPlexGetCellType(dm, p, &ct) >> ablate::checkError;
  switch (ct) {
    case DM_POLYTOPE_TRIANGLE:
//      vof = VOF_2D_Tri(c);
      break;
    case DM_POLYTOPE_QUADRILATERAL:
//      vof = VOF_2D_Quad(c);
      break;
    case DM_POLYTOPE_TETRAHEDRON:
//      vof = VOF_3D_Tetra(c);
      break;
    case DM_POLYTOPE_HEXAHEDRON:
//      vof = VOF_3D_Hex(c);
      break;
    default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No element geometry for cell %" PetscInt_FMT " with type %s", p, DMPolytopeTypes[PetscMax(0, PetscMin(ct, DM_NUM_POLYTOPES))]);
  }


  PetscFree(c);

  return vof;

}

bool LevelSetField::HasInterface(const PetscInt p) {
  bool              hasInterface = false;
  PetscInt          nCells = 0, *cells = NULL;
  PetscInt          i, cStart;
  Vec               phi = LevelSetField::phi;
  const PetscScalar *array;
  PetscScalar       c0;
  DM                dm = LevelSetField::dm;

  DMPlexGetHeightStratum(dm, 0, &cStart, NULL) >> ablate::checkError;

  DMPlexGetNeighborCells(dm, p, 1, -1, -1, PETSC_TRUE, &nCells, &cells) >> ablate::checkError;

  VecGetArrayRead(phi, &array) >> ablate::checkError;
  c0 = array[p - cStart];

  i = 0;
  while (i < nCells && !hasInterface) {
    hasInterface = ((c0 * array[cells[i] - cStart])<=0.0);
    ++i;
  }

  VecRestoreArrayRead(phi, &array) >> ablate::checkError;
  PetscFree(cells) >> ablate::checkError;

  return hasInterface;

}

/* Sphere */
PetscReal LevelSetField::Sphere(PetscReal pos[], PetscReal center[], PetscReal radius) {
  PetscReal shiftedPos[] = {pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]};
  PetscReal phi = PetscSqrtReal(PetscSqr(shiftedPos[0]) + PetscSqr(shiftedPos[1]) + PetscSqr(shiftedPos[2])) - radius;
  return phi;
}

/* Ellipse */
PetscReal LevelSetField::Ellipse(PetscReal pos[], PetscReal center[], PetscReal radius) {
  PetscReal shiftedPos[] = {pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]};
  PetscReal phi = PetscSqr(shiftedPos[0]/0.5) + PetscSqr(shiftedPos[1]/1.25) + PetscSqr(shiftedPos[2]) - radius;
  return phi;
}


/* Star */
PetscReal LevelSetField::Star(PetscReal pos[], PetscReal center[]) {
  PetscReal shiftedPos[] = {pos[0] - center[0], pos[1] - center[1], pos[2] - center[2]};
  PetscReal phi = 400.0*shiftedPos[0]*shiftedPos[0]*shiftedPos[1]*shiftedPos[1]-(1.0-0.5*shiftedPos[0]*shiftedPos[0]-0.5*shiftedPos[1]*shiftedPos[1]);
  return phi;
}




#include "registrar.hpp"
REGISTER(ablate::domain::FieldDescriptor, LevelSetField, "Level Set fields need for interface tracking",
         OPT(ablate::domain::Region, "region", "the region for the level set (defaults to entire domain)"));
