#include <petsc.h>
#include <memory>
#include "domain/RBF/ga.hpp"
#include "domain/RBF/hybrid.hpp"
#include "domain/RBF/imq.hpp"
#include "domain/RBF/mq.hpp"
#include "domain/RBF/phs.hpp"
#include "domain/RBF/rbf.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "environment/runEnvironment.hpp"
#include "utilities/petscUtilities.hpp"



using namespace ablate;


static PetscReal RBFTestFixture_Function(PetscReal x[3], PetscInt dx, PetscInt dy, PetscInt dz) {
    switch (dx + 10 * dy + 100 * dz) {
        case 0:  // Function
            return (1 + sin(4.0 * x[0]) + sin(4.0 * x[1]) + sin(4.0 * x[2]) + cos(x[0] + x[1] + x[2]));
        case 1:  // x
            return (4.0 * cos(4.0 * x[0]) - sin(x[0] + x[1] + x[2]));
        case 2:  // xx
            return (-cos(x[0] + x[1] + x[2]) - 16.0 * sin(4.0 * x[0]));
        case 10:  // y
            return (4.0 * cos(4.0 * x[1]) - sin(x[0] + x[1] + x[2]));
        case 11:  // xy
            return (-cos(x[0] + x[1] + x[2]));
        case 20:  // yy
            return (-cos(x[0] + x[1] + x[2]) - 16.0 * sin(4.0 * x[1]));
        case 100:  // z
            return (4.0 * cos(4.0 * x[2]) - sin(x[0] + x[1] + x[2]));
        case 101:  // xz
            return (-cos(x[0] + x[1] + x[2]));
        case 110:  // yz
            return (-cos(x[0] + x[1] + x[2]));
        case 200:  // zz
            return (-cos(x[0] + x[1] + x[2]) - 16.0 * sin(4.0 * x[2]));
        default:
            throw std::invalid_argument("Unknown derivative of (" + std::to_string(dx) + ", " + std::to_string(dy) + ", " + std::to_string(dz) + ").");
    }
}

void RBFTestFixture_SetData(ablate::solver::Range cellRange, const ablate::domain::Field *field, std::shared_ptr<ablate::domain::SubDomain> subDomain) {
    PetscReal *array, *val, x[3] = {0.0, 0.0, 0.0};
    Vec vec = subDomain->GetVec(*field);
    DM dm = subDomain->GetFieldDM(*field);

    VecGetArray(vec, &array) >> utilities::PetscUtilities::checkError;
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        PetscInt cell = cellRange.points ? cellRange.points[c] : c;
        DMPlexComputeCellGeometryFVM(dm, cell, NULL, x, NULL) >> utilities::PetscUtilities::checkError;
        DMPlexPointLocalFieldRef(dm, cell, field->id, array, &val) >> utilities::PetscUtilities::checkError;
        *val = RBFTestFixture_Function(x, 0, 0, 0);
    }
    VecRestoreArray(vec, &array) >> utilities::PetscUtilities::checkError;
}


int main(int argc, char** argv) {



  environment::RunEnvironment::Initialize(&argc, &argv);
  utilities::PetscUtilities::Initialize();

  std::vector<std::shared_ptr<domain::rbf::RBF>> rbfList = {std::make_shared<ablate::domain::rbf::GA>(4, 3.888078956798655e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::MQ>(4, 3.888078956798655e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::IMQ>(4, 3.888078956798655e-02, false, false),
                                                            std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false),
                                                            std::make_shared<ablate::domain::rbf::HYBRID>(4, std::vector<double>{1.0, 0.001},
                                                                                                          std::vector<std::shared_ptr<ablate::domain::rbf::RBF>>{
                                                                                                              std::make_shared<ablate::domain::rbf::GA>(4, 3.888078956798655e-02, false, false),
                                                                                                              std::make_shared<ablate::domain::rbf::PHS>(4, 2, false, false)},
                                                                                                          false, false)};

  //             Make the field
  std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptor = {
      std::make_shared<ablate::domain::FieldDescription>("fieldA", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM)};

  //             Create the mesh
  //      Note that using -dm_view :mesh.tex:ascii_latex -dm_plex_view_scale 10 -dm_plex_view_numbers_depth 1,0,1 will create a mesh, changing numbers_depth as appropriate
  auto mesh = std::make_shared<domain::BoxMesh>("mesh", fieldDescriptor, std::vector<std::shared_ptr<domain::modifiers::Modifier>>{std::make_shared<domain::modifiers::DistributeWithGhostCells>(3)},
                                                std::vector<int>{21, 21, 21}, std::vector<double>{-1.0, -1.0, -1.0}, std::vector<double>{1.0 ,1.0, 1.0}, std::vector<std::string>{}, true);

  mesh->InitializeSubDomains();

  std::shared_ptr<ablate::domain::SubDomain> subDomain = mesh->GetSubDomain(domain::Region::ENTIREDOMAIN);

  // The field containing the data
  const ablate::domain::Field *field = &(subDomain->GetField("fieldA"));

  ablate::solver::Range cellRange;
  for (unsigned long int j = 0; j < rbfList.size(); ++j) {
      rbfList[j]->Setup(subDomain);  // This causes issues (I think)

      //         Initialize
      rbfList[j]->GetCellRange(subDomain, nullptr, cellRange);
      rbfList[j]->Initialize(cellRange);
      rbfList[j]->RestoreRange(cellRange);
  }

  // Now set the data using the first RBF. All will use the same data
  rbfList[0]->GetCellRange(subDomain, nullptr, cellRange);
  RBFTestFixture_SetData(cellRange, field, subDomain);

  // Now check derivatives
  std::vector<PetscInt> dx = {0, 1, 2, 0, 1, 0, 0, 1, 0, 0}, dy = {0, 0, 0, 1, 1, 2, 0, 0, 1, 0}, dz = {0, 0, 0, 0, 0, 0, 1, 1, 1, 2};
//  std::vector<PetscReal> maxErrorList = {1e-15, 3.5e-05, 3.5e-03, 5.6e-05, 1.7e-04, 3.0e-03, 1.51e-04, 6.3e-03, 5.9e-03, 2.1e-03};
  PetscInt c, cell;
//  PetscReal maxError;
  PetscReal x[3];
  PetscReal err = -1.0;
  PetscReal val;
  DM dm = subDomain->GetDM();

  for (unsigned long int i = 0; i < dx.size(); ++i) {  // Iterate over each of the requested derivatives

        // 3D results take too long to run, so just check a corner
        for (unsigned long int j = 0; j < rbfList.size(); ++j) {  // Check each RBF
            c = 0;

            cell = cellRange.points ? cellRange.points[c] : c;
            val = rbfList[j]->EvalDer(field, c, dx[i], dy[i], dz[i]);

            DMPlexComputeCellGeometryFVM(dm, cell, NULL, x, NULL) >> ablate::utilities::PetscUtilities::checkError;
            err = PetscAbsReal(val - RBFTestFixture_Function(x, dx[i], dy[i], dz[i]));
            printf("%e\n", err);

      }

    }


  rbfList[0]->RestoreRange(cellRange);

  //      ablate::environment::RunEnvironment::Finalize();


}
