#include <numeric>
#include "domain/boxMesh.hpp"
#include "domain/constFieldAccessor.hpp"
#include "domain/fieldAccessor.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "petscTestFixture.hpp"

class FieldAccessorTestFixture : public testingResources::PetscTestFixture {};

TEST_F(FieldAccessorTestFixture, ShouldReadAndEditAccessors) {
    // ARRANGE
    // define a test field to compute gradients
    std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {
        std::make_shared<ablate::domain::FieldDescription>("fieldA", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM),
        std::make_shared<ablate::domain::FieldDescription>("fieldB", "", std::vector<std::string>{"alpha", "beta", "gamma"}, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM),
        std::make_shared<ablate::domain::FieldDescription>("fieldC", "", std::vector<std::string>{"alpha", "beta", "gamma"}, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM),
        std::make_shared<ablate::domain::FieldDescription>("auxFieldA", "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM),
        std::make_shared<ablate::domain::FieldDescription>("auxFieldB", "", std::vector<std::string>{"alpha", "beta", "gamma"}, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM),
        std::make_shared<ablate::domain::FieldDescription>("auxFieldC", "", std::vector<std::string>{"alpha", "beta", "gamma"}, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM)};

    // create a zeroD domain for testing
    auto domain = std::make_shared<ablate::domain::BoxMesh>(
        "testMesh", fieldDescriptors, std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{}, std::vector<int>{5, 6}, std::vector<double>{0.0, 0.0}, std::vector<double>{1.0, 1.0});

    domain->InitializeSubDomains();

    // Setup the fields using standard init
    auto subDomain = domain->GetSubDomain(nullptr);
    auto fieldFunctions = {
        std::make_shared<ablate::mathFunctions::FieldFunction>("fieldA", ablate::mathFunctions::Create("x")),
        std::make_shared<ablate::mathFunctions::FieldFunction>("fieldB", ablate::mathFunctions::Create("x, y, x*y")),
    };
    domain->ProjectFieldFunctions(fieldFunctions, domain->GetSolutionVector());

    auto auxFieldFunctions = {
        std::make_shared<ablate::mathFunctions::FieldFunction>("auxFieldA", ablate::mathFunctions::Create("10*x")),
        std::make_shared<ablate::mathFunctions::FieldFunction>("auxFieldB", ablate::mathFunctions::Create("x*10, y*10, x*y*10")),
    };
    subDomain->ProjectFieldFunctionsToLocalVector(auxFieldFunctions, subDomain->GetAuxVector());

    // ACT
    auto fieldAAccessor = subDomain->GetConstSolutionAccessor("fieldA");
    auto fieldBAccessor = subDomain->GetConstSolutionAccessor("fieldB");
    auto auxFieldAAccessor = subDomain->GetConstAuxAccessor("auxFieldA");
    auto auxFieldBAccessor = subDomain->GetConstAuxAccessor("auxFieldB");
    auto fieldCAccessor = subDomain->GetSolutionAccessor("fieldC");
    auto auxFieldCAccessor = subDomain->GetAuxAccessor("auxFieldC");

    // ASSERT march over every cell to make sure they are correct and we can set them
    ablate::domain::Range cellRange;
    subDomain->GetCellRange(nullptr, cellRange);

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        auto cell = cellRange.GetPoint(c);

        // Compute the cell location
        PetscReal centroid[3];
        DMPlexComputeCellGeometryFVM(subDomain->GetDM(), cell, nullptr, centroid, nullptr);

        auto fieldA = fieldAAccessor[cell];
        ASSERT_DOUBLE_EQ(fieldA[0], centroid[0]);

        auto fieldB = fieldBAccessor[cell];
        ASSERT_DOUBLE_EQ(fieldB[0], centroid[0]);
        ASSERT_DOUBLE_EQ(fieldB[1], centroid[1]);
        ASSERT_DOUBLE_EQ(fieldB[2], centroid[0] * centroid[1]);

        auto auxFieldA = auxFieldAAccessor[cell];
        ASSERT_DOUBLE_EQ(auxFieldA[0], centroid[0] * 10);

        auto auxFieldB = auxFieldBAccessor[cell];
        ASSERT_DOUBLE_EQ(auxFieldB[0], centroid[0] * 10);
        ASSERT_DOUBLE_EQ(auxFieldB[1], centroid[1] * 10);
        ASSERT_DOUBLE_EQ(auxFieldB[2], centroid[0] * centroid[1] * 10);

        // Set field C to B
        for (PetscInt d = 0; d < fieldCAccessor.GetField().numberComponents; ++d) {
            fieldCAccessor[cell][d] = fieldB[d];
        }
        for (PetscInt d = 0; d < auxFieldCAccessor.GetField().numberComponents; ++d) {
            auxFieldCAccessor[cell][d] = auxFieldB[d];
        }
    }

    subDomain->RestoreRange(cellRange);

    // Now make sure that the vecs are the same
    IS fieldBIS;
    Vec fieldBVec;
    DM fieldBDm;
    subDomain->GetFieldGlobalVector(subDomain->GetField("fieldB"), &fieldBIS, &fieldBVec, &fieldBDm);
    IS fieldCIS;
    Vec fieldCVec;
    DM fieldCDm;
    subDomain->GetFieldGlobalVector(subDomain->GetField("fieldC"), &fieldCIS, &fieldCVec, &fieldCDm);
    PetscBool fieldBEqualFieldC;
    VecEqual(fieldBVec, fieldCVec, &fieldBEqualFieldC);
    ASSERT_TRUE(fieldBEqualFieldC);

    subDomain->RestoreFieldGlobalVector(subDomain->GetField("fieldA"), &fieldBIS, &fieldBVec, &fieldBDm);
    subDomain->RestoreFieldGlobalVector(subDomain->GetField("fieldC"), &fieldCIS, &fieldCVec, &fieldCDm);

    // Now make sure that the vecs are the same
    IS auxFieldBIS;
    Vec auxFieldBVec;
    DM auxFieldBDm;
    subDomain->GetFieldLocalVector(subDomain->GetField("auxFieldB"), 0.0, &auxFieldBIS, &auxFieldBVec, &auxFieldBDm);
    IS auxFieldCIS;
    Vec auxFieldCVec;
    DM auxFieldCDm;
    subDomain->GetFieldLocalVector(subDomain->GetField("auxFieldC"), 0.0, &auxFieldCIS, &auxFieldCVec, &auxFieldCDm);
    VecEqual(auxFieldBVec, auxFieldCVec, &fieldBEqualFieldC);
    ASSERT_TRUE(fieldBEqualFieldC);

    subDomain->RestoreFieldLocalVector(subDomain->GetField("auxFieldA"), &auxFieldBIS, &auxFieldBVec, &auxFieldBDm);
    subDomain->RestoreFieldLocalVector(subDomain->GetField("auxFieldC"), &auxFieldCIS, &auxFieldCVec, &auxFieldCDm);
}
