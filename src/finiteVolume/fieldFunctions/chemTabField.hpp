#ifndef ABLATELIBRARY_FIELD_FUNCTION_CHEMTABFIELD_HPP
#define ABLATELIBRARY_FIELD_FUNCTION_CHEMTABFIELD_HPP

#include <eos/eos.hpp>
#include <mathFunctions/fieldFunction.hpp>
#include "compressibleFlowState.hpp"
#include "eos/chemTab.hpp"

namespace ablate::finiteVolume::fieldFunctions {

/**
 * Class that species progress variable from a chemTab model
 */
class ChemTabField : public ablate::mathFunctions::FieldFunction {
   private:
    /**
     * Hold the static progress variables
     */
    std::vector<PetscReal> progressVariables;

    /**
     * private function to compute chemtab progress variables
     * @param dim
     * @param time
     * @param x
     * @param Nf
     * @param u
     * @param ctx
     * @return
     */
    static PetscErrorCode ComputeChemTabProgress(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

   public:
    /**
     * Determines the progress field for initialization
     * @param initializer
     * @param eos
     */
    explicit ChemTabField(const std::string& initializer, std::shared_ptr<ablate::eos::EOS> eos);
};

}  // namespace ablate::finiteVolume::fieldFunctions
#endif  // ABLATELIBRARY_FIELD_FUNCTION_CHEMTABFIELD_HPP
