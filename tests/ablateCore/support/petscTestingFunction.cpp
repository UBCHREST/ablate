#include "petscTestingFunction.hpp"

tests::ablateCore::support::PetscTestingFunction::PetscTestingFunction(std::string formula) {
    // define the x,y,z and t variables
    parser.DefineVar("x", &coordinate[0]);
    parser.DefineVar("y", &coordinate[1]);
    parser.DefineVar("z", &coordinate[2]);
    parser.DefineVar("t", &time);

    parser.SetExpr(formula);
    parser.Eval();
}

PetscErrorCode tests::ablateCore::support::PetscTestingFunction::ApplySolution(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    try {
        auto parser = (PetscTestingFunction *)ctx;

        // update the coordinates
        parser->coordinate[0] = 0;
        parser->coordinate[1] = 0;
        parser->coordinate[2] = 0;

        for (auto i = 0; i < std::min(dim, 3); i++) {
            parser->coordinate[i] = x[i];
        }
        parser->time = time;

        // Evaluate
        int functionSize = 0;
        auto rawResult = parser->parser.Eval(functionSize);

        // copy over
        for (auto i = 0; i < Nf; i++) {
            u[i] = rawResult[i];
        }

    } catch (std::exception &exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exception.what());
    }
    PetscFunctionReturn(0);
}

PetscErrorCode tests::ablateCore::support::PetscTestingFunction::ApplySolutionTimeDerivative(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    try {
        auto parser = (PetscTestingFunction *)ctx;

        // update the coordinates
        parser->coordinate[0] = 0;
        parser->coordinate[1] = 0;
        parser->coordinate[2] = 0;

        for (auto i = 0; i < std::min(dim, 3); i++) {
            parser->coordinate[i] = x[i];
        }
        parser->time = time;

        // Evaluate
        int functionSize = 0;
        auto rawResult = parser->parser.Eval(functionSize);

        // copy over
        for (auto i = 0; i < Nf; i++) {
            u[i] = rawResult[i + Nf];
        }

    } catch (std::exception &exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exception.what());
    }
    PetscFunctionReturn(0);
}
