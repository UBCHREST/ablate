#ifndef ABLATELIBRARY_CHEMISTRYMODEL_HPP
#define ABLATELIBRARY_CHEMISTRYMODEL_HPP

#include <petsc.h>
#include <string>
#include <vector>

namespace ablate::chemistry {
class ChemistryModel {
   public:
    virtual ~ChemistryModel() = default;

    /**
     * Function pointer allowing the computing of mass functions from progress variables
     */
    using ComputeMassFractionsFunction = void (*)(const PetscReal progressVariables[], const std::size_t progressVariablesSize, PetscReal* massFractions, const std::size_t massFractionsSize, void* ctx);
    /**
     * Function pointer allowing the computing of energy*density source function and functions for each progress variable
     */
    using ComputeSourceFunction = void (*)(const PetscReal progressVariables[], const std::size_t progressVariablesSize, PetscReal ZMix, PetscReal *predictedSourceEnergy, PetscReal* progressVariableSource, const std::size_t progressVariableSourceSize, void* ctx);

    /**
     * Returns a vector of all species required for this model.  The species order indicates the correct order for other functions
     * @return
     */
    virtual const std::vector<std::string>& GetSpecies() const = 0;
    /**
     * Returns a vector of all progress variables (including zMix) required for this model.  The progress variable order indicates the correct order for other functions
     * @return
     */
    virtual const std::vector<std::string>& GetProgressVariables() const = 0;

    /**
     * Computes the progresses variables for a given mass fraction
     * @return
     */
    virtual void ComputeProgressVariables(const PetscReal massFractions[], const std::size_t massFractionsSize, PetscReal* progressVariables, const std::size_t progressVariablesSize) const = 0;

    /**
     * Support functions to get access to c-style pointer functions
     * @return
     */
    virtual ComputeMassFractionsFunction GetComputeMassFractionsFunction() = 0;
    virtual ComputeSourceFunction GetComputeSourceFunction() = 0;
    virtual void* GetContext() { return nullptr; }
};
}  // namespace ablate::chemistry

#endif  // ABLATELIBRARY_CHEMISTRYMODEL_HPP
