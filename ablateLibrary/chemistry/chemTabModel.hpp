#ifndef ABLATELIBRARY_CHEMTABMODEL_HPP
#define ABLATELIBRARY_CHEMTABMODEL_HPP

#include <tensorflow/c/c_api.h>
#include <filesystem>
#include <istream>
#include <petscmat.h>
#include "chemistryModel.hpp"

namespace ablate::chemistry {

#ifdef WITH_TENSORFLOW
class ChemTabModel : public ChemistryModel {
   private:
    TF_Graph* graph = nullptr;
    TF_Status* status = nullptr;
    TF_SessionOptions* sessionOpts = nullptr;
    TF_Buffer* runOpts = nullptr;
    TF_Session* session = nullptr;
    std::vector<std::string> speciesNames = std::vector<std::string>(0);
    std::vector<std::string> progressVariablesNames = std::vector<std::string>(0);

    PetscReal** Wmat = nullptr;
    PetscReal** iWmat = nullptr;
    /**
     * private implementations of support functions
     */
    static void ChemTabModelComputeMassFractionsFunction(const PetscReal progressVariables[], const std::size_t progressVariablesSize, PetscReal* massFractions, const std::size_t massFractionsSize, void* ctx);
    static void ChemTabModelComputeSourceFunction(const PetscReal progressVariables[], const std::size_t progressVariablesSize, PetscReal ZMix, PetscReal *predictedSourceEnergy, PetscReal* progressVariableSource, const std::size_t progressVariableSourceSize, void* ctx);
    void ExtractMetaData(std::istream& inputStream);
    void LoadBasisVectors(std::istream& inputStream, std::size_t columns, PetscReal** W);


   public:
    explicit ChemTabModel(std::filesystem::path path);
    ~ChemTabModel() override;

    /**
     * Returns a vector of all species required for this model.  The species order indicates the correct order for other functions
     * @return
     */
    const std::vector<std::string>& GetSpecies() const override;

    /**
     * Returns a vector of all progress variables (including zMix) required for this model.  The progress variable order indicates the correct order for other functions
     * @return
     */
    const std::vector<std::string>& GetProgressVariables() const override;

    /**
     * Computes the progresses variables for a given mass fraction
     * @return
     */
    void ComputeProgressVariables(const PetscReal massFractions[], const std::size_t massFractionsSize, PetscReal* progressVariables, const std::size_t progressVariablesSize) const override;

    /**
     * Support functions to get access to c-style pointer functions
     * @return
     */
    ComputeMassFractionsFunction GetComputeMassFractionsFunction() override { return ChemTabModelComputeMassFractionsFunction; }
    ComputeSourceFunction GetComputeSourceFunction() override { return ChemTabModelComputeSourceFunction; }
    void* GetContext() override { return this; }
};

#else
class ChemTabModel : public ChemistryModel {
   public:
    static inline const std::string errorMessage = "Using the ChemTabModel requires Tensorflow to be compile with ABLATE.";
    ChemTabModel(std::filesystem::path path) { throw std::runtime_error(errorMessage); }

    const std::vector<std::string>& GetSpecies() const override { throw std::runtime_error(errorMessage); }
    const std::vector<std::string>& GetProgressVariables() const override { throw std::runtime_error(errorMessage); }

    void ComputeProgressVariables(const PetscReal massFractions[], PetscReal* progressVariables) const override { throw std::runtime_error(errorMessage); }

    ComputeMassFractionsFunction GetComputeMassFractionsFunction() override { throw std::runtime_error(errorMessage); }
    ComputeSourceFunction GetComputeSourceFunction() override { throw std::runtime_error(errorMessage); }
};
#endif
}  // namespace ablate::chemistry
#endif  // ABLATELIBRARY_CHEMTABMODEL_HPP
