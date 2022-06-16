
#include "arbitrarySource.hpp"
ablate::finiteVolume::processes::ArbitrarySource::ArbitrarySource(std::map<std::string, std::shared_ptr<ablate::mathFunctions::MathFunction>> functions) : functions(std::move(functions)) {}

void ablate::finiteVolume::processes::ArbitrarySource::Initialize(ablate::finiteVolume::FiniteVolumeSolver &fvmSolver) {
    for (const auto &[fieldName, function] : functions) {
        // Get the field from the subDomain
        const auto &field = fvmSolver.GetSubDomain().GetField(fieldName);

        petscFunctions.emplace_back(PetscFunctionStruct{.petscFunction = function->GetPetscFunction(), .petscContext = function->GetContext(), .fieldSize = field.numberComponents});

        // add the source function
        fvmSolver.RegisterRHSFunction(ComputeArbitrarySource, &petscFunctions.back(), {fieldName}, {}, {});
    }
}

PetscErrorCode ablate::finiteVolume::processes::ArbitrarySource::ComputeArbitrarySource(PetscInt dim, PetscReal time, const PetscFVCellGeom *cg, const PetscInt *uOff, const PetscScalar *u,
                                                                                        const PetscInt *aOff, const PetscScalar *a, PetscScalar *f, void *ctx) {
    PetscFunctionBegin;
    auto function = (PetscFunctionStruct *)ctx;
    PetscErrorCode ierr = function->petscFunction(dim, time, cg->centroid, function->fieldSize, f, function->petscContext);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
#define COMMA ,
REGISTER_PASS_THROUGH(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::ArbitrarySource, "uses math functions to add arbitrary sources to the fvm method",
                      std::map<std::string COMMA ablate::mathFunctions::MathFunction>);
