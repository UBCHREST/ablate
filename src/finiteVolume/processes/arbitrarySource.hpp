#ifndef ABLATELIBRARY_ARBITRARYSOURCE_HPP
#define ABLATELIBRARY_ARBITRARYSOURCE_HPP

#include "process.hpp"
namespace ablate::finiteVolume::processes {
/**
 * This class uses math functions to add arbitrary sources to the fvm method
 */
class ArbitrarySource : public Process {
    //! list of functions used to compute the arbitrary source
    const std::map<std::string, std::shared_ptr<ablate::mathFunctions::MathFunction>> functions;

    /**
     * private function to compute  source
     * @return
     */
    static PetscErrorCode ComputeArbitrarySource(PetscInt dim, PetscReal time, const PetscFVCellGeom* cg, const PetscInt uOff[], const PetscScalar u[], const PetscInt aOff[], const PetscScalar a[],
                                                 PetscScalar f[], void* ctx);

    //! pre store the petsc function and context
    struct PetscFunctionStruct {
        mathFunctions::PetscFunction petscFunction;
        void* petscContext;
        PetscInt fieldSize;
    };

    //! Store pointers to the petsc functions
    std::vector<PetscFunctionStruct> petscFunctions;

   public:
    explicit ArbitrarySource(std::map<std::string, std::shared_ptr<ablate::mathFunctions::MathFunction>> functions);

    /**
     * public function to link this process with the fvm solver
     * @param flow
     */
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver& fvmSolver) override;
};

}  // namespace ablate::finiteVolume::processes

#endif  // ABLATELIBRARY_ARBITRARYSOURCE_HPP
