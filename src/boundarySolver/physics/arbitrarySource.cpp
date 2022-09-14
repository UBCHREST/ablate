#include "arbitrarySource.hpp"
#include "utilities/vectorUtilities.hpp"
ablate::boundarySolver::physics::ArbitrarySource::ArbitrarySource(std::map<std::string, std::shared_ptr<ablate::mathFunctions::MathFunction>> functions,
                                                                  ablate::boundarySolver::BoundarySolver::BoundarySourceType boundarySourceType)
    :

      functions(functions),
      boundarySourceType(boundarySourceType) {}

void ablate::boundarySolver::physics::ArbitrarySource::Initialize(ablate::boundarySolver::BoundarySolver &bSolver) {
    // Build the list of output components
    std::vector<std::string> sourceComponents;
    for (const auto &function : functions) {
        sourceComponents.push_back(function.first);
    }

    bSolver.RegisterFunction(ArbitrarySourceFunction, this, sourceComponents, std::vector<std::string>{}, std::vector<std::string>{}, boundarySourceType);

    bSolver.RegisterPreStep([this](auto ts, auto &solver) {
        PetscFunctionBeginUser;
        PetscErrorCode ierr = TSGetTime(ts, &(this->currentTime));
        CHKERRQ(ierr);

        PetscFunctionReturn(0);
    });
}

PetscErrorCode ablate::boundarySolver::physics::ArbitrarySource::ArbitrarySourceFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg,
                                                                                         const PetscFVCellGeom *boundaryCell, const PetscInt *uOff, const PetscScalar *boundaryValues,
                                                                                         const PetscScalar **stencilValues, const PetscInt *aOff, const PetscScalar *auxValues,
                                                                                         const PetscScalar **stencilAuxValues, PetscInt stencilSize, const PetscInt *stencil,
                                                                                         const PetscScalar *stencilWeights, const PetscInt *sOff, PetscScalar *source, void *ctx) {
    PetscFunctionBeginUser;
    auto arbitrarySource = (ablate::boundarySolver::physics::ArbitrarySource *)ctx;

    // Keep track of the offset
    PetscInt functionCount = 0;
    for (const auto &function : arbitrarySource->functions) {
        source[sOff[functionCount++]] = function.second->Eval(fg->centroid, dim, arbitrarySource->currentTime);
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
#define COMMA ,
REGISTER(ablate::boundarySolver::BoundaryProcess, ablate::boundarySolver::physics::ArbitrarySource, "uses math functions to add arbitrary sources to the boundary solver",
         ARG(std::map<std::string COMMA ablate::mathFunctions::MathFunction>, "functions", "the map/object of names/functions"),
         ARG(EnumWrapper<ablate::boundarySolver::BoundarySolver::BoundarySourceType>, "type", "the boundary source type (Point, Distributed, Flux, Face"));