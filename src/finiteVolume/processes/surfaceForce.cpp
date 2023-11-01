#include "domain/RBF/phs.hpp"
#include "surfaceForce.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "levelSet/levelSetUtilities.hpp"
#include "registrar.hpp"
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscSupport.hpp"
#include "mathFunctions/functionFactory.hpp"

#define xexit(S, ...) {PetscFPrintf(MPI_COMM_WORLD, stderr, \
  "\x1b[1m(%s:%d, %s)\x1b[0m\n  \x1b[1m\x1b[90mexiting:\x1b[0m " S "\n", \
  __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); exit(0);}
ablate::finiteVolume::processes::SurfaceForce::SurfaceForce(PetscReal sigma) : sigma(sigma) {
    printf("Sigma is equal to %e\n", sigma);
}

// Done once at the beginning of every run
void ablate::finiteVolume::processes::SurfaceForce::Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) {
    flow.RegisterRHSFunction(ComputeSource, this);
}

std::shared_ptr<ablate::domain::SubDomain> subDomain = nullptr;

// Called every time the mesh changes
void ablate::finiteVolume::processes::SurfaceForce::Initialize(ablate::finiteVolume::FiniteVolumeSolver &solver) {
  subDomain = solver.GetSubDomainPtr();
}



inline PetscReal SmoothDirac(PetscReal c, PetscReal c0, PetscReal t) {
  return (PetscAbsReal(c-c0) < t ? 0.5*(1.0 + cos(M_PI*(c - c0)/t))/t : 0);
}


//#include "utilities/mpiUtilities.hpp"
//static void SaveCellData(const char fname[255], Vec vec, const ablate::domain::Field *field, PetscInt Nc, std::shared_ptr<ablate::domain::SubDomain> subDomain) {

//  ablate::domain::Range range;
//  PetscReal    *array, *val;
//  DM            dm  = subDomain->GetFieldDM(*field);
//  PetscInt      dim = subDomain->GetDimensions();
//  MPI_Comm      comm = PetscObjectComm((PetscObject)dm);
//  int rank, size;
//  MPI_Comm_size(comm, &size) >> ablate::utilities::MpiUtilities::checkError;
//  MPI_Comm_rank(comm, &rank) >> ablate::utilities::MpiUtilities::checkError;

//  subDomain->GetCellRange(nullptr, range);

//  VecGetArray(vec, &array) >> ablate::utilities::PetscUtilities::checkError;



//  for (PetscInt r = 0; r < size; ++r) {
//    if ( rank==r ) {

//      FILE *f1;
//      if ( rank==0 ) f1 = fopen(fname, "w");
//      else f1 = fopen(fname, "a");

//      for (PetscInt c = range.start; c < range.end; ++c) {
//        PetscInt cell = range.points ? range.points[c] : c;

//        if (ablate::levelSet::Utilities::ValidCell(dm, cell)) {

//          PetscReal x0[3];
//          DMPlexComputeCellGeometryFVM(dm, cell, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;
//          DMPlexPointLocalFieldRef(dm, cell, field->id, array, &val) >> ablate::utilities::PetscUtilities::checkError;

//          for (PetscInt d = 0; d < dim; ++d) {
//            fprintf(f1, "%+f\t", x0[d]);
//          }

//          for (PetscInt i = 0; i < Nc; ++i) {
//            fprintf(f1, "%+f\t", val[i]);
//          }
//          fprintf(f1, "\n");
//        }
//      }
//      fclose(f1);
//    }

//    MPI_Barrier(PETSC_COMM_WORLD);
//  }


//  VecRestoreArray(vec, &array) >> ablate::utilities::PetscUtilities::checkError;
//  ablate::domain::RestoreRange(range);
//}


PetscErrorCode ablate::finiteVolume::processes::SurfaceForce::ComputeSource(const FiniteVolumeSolver &flow, DM dm, PetscReal time, Vec locX, Vec locF, void *ctx) {
    PetscFunctionBegin;

    ablate::finiteVolume::processes::SurfaceForce *process = (ablate::finiteVolume::processes::SurfaceForce *)ctx;
//    std::shared_ptr<ablate::domain::SubDomain> subDomain = this->subDomainPTR;
    const PetscInt dim = subDomain->GetDimensions();
    ablate::domain::Range cellRange;

    const ablate::domain::Field *vofField = &(subDomain->GetField(TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD));
    const ablate::domain::Field *lsField = &(subDomain->GetField("levelSet"));
    const ablate::domain::Field *vertexNormalField = &(subDomain->GetField("vertexNormal"));
    const ablate::domain::Field *curvField = &(subDomain->GetField("curvature"));
    const ablate::domain::Field *cellNormalField = &(subDomain->GetField("cellNormal"));
    const ablate::domain::Field *eulerField = &(subDomain->GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD));


    DM eulerDM = subDomain->GetFieldDM(*eulerField); // Get an euler-specific DM in case it's not in the same solution vector as the VOF field
    const PetscReal sigma = process->sigma; // Surface tension coefficient

    PetscScalar *fArray = nullptr;
    const PetscScalar *auxArray = nullptr, *xArray = nullptr;

    // The grid spacing
    PetscReal h;
    DMPlexGetMinRadius(dm, &h) >> ablate::utilities::PetscUtilities::checkError;
    h *= 2.0; // Min radius returns the distance between a cell-center and a face. Double it to get the average cell size
//SaveCellData("vof0.txt", locX, vofField, 1, subDomain);

    ablate::levelSet::Utilities::Reinitialize(subDomain, locX, vofField, 5, lsField, vertexNormalField, cellNormalField, curvField);

    DM auxDM = subDomain->GetAuxDM();
    Vec auxVec = subDomain->GetAuxVector();


    VecGetArray(locF, &fArray) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

    subDomain->GetCellRange(nullptr, cellRange);
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
      PetscInt cell = cellRange.GetPoint(c);

      PetscReal cellPhi = 0.0, dirac = -1.0;

      if (ablate::levelSet::Utilities::ValidCell(auxDM, cell)) {

        PetscInt nv, *verts;
        DMPlexCellGetVertices(auxDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
        for (PetscInt v = 0; v < nv; ++v) {
          const PetscReal *phi = nullptr;
          xDMPlexPointLocalRead(auxDM, verts[v], lsField->id, auxArray, &phi);

          cellPhi += *phi;
        }
        cellPhi /= nv;
        DMPlexCellRestoreVertices(auxDM, cell, &nv, &verts) >> ablate::utilities::PetscUtilities::checkError;
        dirac = SmoothDirac(cellPhi, 0.0, 2.0*h);
      }

      PetscScalar *eulerSource = nullptr;
      xDMPlexPointLocalRef(eulerDM, cell, eulerField->id, fArray, &eulerSource) >> utilities::PetscUtilities::checkError;

      // Start by zeroing out everything
      eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHO] = 0.0;
      eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] = 0.0;
      for (PetscInt d = 0; d < dim; ++d) {
          eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = 0.0;
      }

      if (dirac > 1e-10){
        // Normal at the cell-center
        PetscReal *n = nullptr;
        xDMPlexPointLocalRead(auxDM, cell, cellNormalField->id, auxArray, &n);

        // Curvature at the cell-center
        PetscReal *H = nullptr;
        xDMPlexPointLocalRead(auxDM, cell, curvField->id, auxArray, &H);

        const PetscScalar *euler = nullptr;
        xDMPlexPointLocalRead(eulerDM, cell, eulerField->id, xArray, &euler) >> utilities::PetscUtilities::checkError;
        const PetscScalar density = euler[ablate::finiteVolume::CompressibleFlowFields::RHO];

        for (PetscInt d = 0; d < dim; ++d) {
            // calculate surface force and energy

            PetscReal surfaceForce = -dirac* density * sigma * H[0] * n[d];
            PetscReal vel = euler[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] / density;
            PetscReal surfaceEnergy = surfaceForce * vel;

            // add in the contributions
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOU + d] = surfaceForce;
            eulerSource[ablate::finiteVolume::CompressibleFlowFields::RHOE] += surfaceEnergy;
        }
      }


    }
    subDomain->RestoreRange(cellRange);
//SaveCellData("locF.txt", locF, eulerField, 5, subDomain);
//exit(0);

    // Cleanup
    VecRestoreArray(locF, &fArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(auxVec, &auxArray) >> ablate::utilities::PetscUtilities::checkError;

    PetscFunctionReturn(PETSC_SUCCESS);
}

ablate::finiteVolume::processes::SurfaceForce::~SurfaceForce() {

}

REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SurfaceForce, "calculates surface tension force and adds source terms",
         ARG(PetscReal, "sigma", "sigma, surface tension coefficient"));
