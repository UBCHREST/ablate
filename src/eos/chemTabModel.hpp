#ifndef ABLATELIBRARY_CHEMTABMODEL_HPP
#define ABLATELIBRARY_CHEMTABMODEL_HPP

#include <petscmat.h>
#include <filesystem>
#include <istream>
#include "chemistryModel.hpp"
#ifdef WITH_TENSORFLOW
#include <tensorflow/c/c_api.h>
#include "utilities/vectorUtilities.hpp"
#endif

namespace ablate::eos {

#ifdef WITH_TENSORFLOW
class ChemTabModel : public ChemistryModel {
   private:
    //! use the reference eos to compute properties from the decoded progressVariables to yi
    std::shared_ptr<ablate::eos::EOS> referenceEOS;

    // hold the required tensorflow information
    TF_Graph* graph = nullptr;
    TF_Status* status = nullptr;
    TF_SessionOptions* sessionOpts = nullptr;
    TF_Buffer* runOpts = nullptr;
    TF_Session* session = nullptr;
    std::vector<std::string> speciesNames = std::vector<std::string>(0);
    std::vector<std::string> progressVariablesNames = std::vector<std::string>(0);

    PetscReal** Wmat = nullptr;
    PetscReal** iWmat = nullptr;
    PetscReal* sourceEnergyScaler = nullptr;

    /**
     * private implementations of support functions
     */
    void ExtractMetaData(std::istream& inputStream);
    void LoadBasisVectors(std::istream& inputStream, std::size_t columns, PetscReal** W);

    /**
     * Private function to compute the chemistry source given the density, energy, and progress variable offset
     * @param fields
     * @param conserved
     * @param source
     */
    void ChemistrySource(PetscInt densityOffset, PetscInt energyOffset, PetscInt progressVariableOffset, const PetscReal conserved[], PetscReal* source) const;

   public:
    explicit ChemTabModel(std::filesystem::path path);
    ~ChemTabModel() override;

    /**
     * As far as other parts of the code is concerned the chemTabEos does not expect species
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetSpecies() const override { return ablate::utilities::VectorUtilities::Empty<std::string>; }

    /**
     * return the reference species used for the underlying eos to generate the progress variables
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetReferenceSpecies() const { return speciesNames; }

    /**
     * As far as other parts of the code is concerned the chemTabEos does not expect species
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetExtraVariables() const override { return progressVariablesNames; }

    /**
     * Single function to produce ChemistryFunction calculator based upon the available fields and sources.
     * @param fields in the conserved/source arrays
     * @param property
     * @param fields
     * @return
     */
    std::shared_ptr<SourceCalculator> CreateSourceCalculator(const std::vector<domain::Field>& fields, const solver::Range& cellRange) override { return nullptr; }

    /**
     *
     * @param fields
     * @param dt
     * @param conserved
     * @param source
     */
    void ChemistrySource(const std::vector<domain::Field>& fields, PetscReal dt, const PetscReal conserved[], PetscReal* source) const;

    /**
     * helper function to compute the progress variables from the mass fractions
     * @param massFractions
     * @param massFractionsSize
     * @param progressVariables
     * @param progressVariablesSize
     */
    void ComputeProgressVariables(const PetscReal* massFractions, std::size_t massFractionsSize, PetscReal* progressVariables, std::size_t progressVariablesSize) const;

    /**
     * helper function to compute the mass fractions = from the mass fractions progress variables
     * @param massFractions
     * @param massFractionsSize
     * @param progressVariables
     * @param progressVariablesSize
     */
    void ComputeMassFractions(const PetscReal* progressVariables, std::size_t progressVariablesSize, PetscReal* massFractions, std::size_t massFractionsSize);

    /**
     * Print the details of this eos
     * @param stream
     */
    void View(std::ostream& stream) const override;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] eos::ThermodynamicFunction GetThermodynamicFunction(eos::ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override { return {}; }

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] eos::ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override { return {}; }

    /**
     * Single function to produce fieldFunction function for any two properties, velocity, and species mass fractions.  These calls can be slower and should be used for init/output only
     * @param field
     * @param property1
     * @param property2
     */
    [[nodiscard]] eos::FieldFunction GetFieldFunctionFunction(const std::string& field, eos::ThermodynamicProperty property1, eos::ThermodynamicProperty property2) const override { return {}; }
};

#else
class ChemTabModel : public ChemistryModel {
   public:
    static inline const std::string errorMessage = "Using the ChemTabModel requires Tensorflow to be compile with ABLATE.";
    ChemTabModel(std::filesystem::path path) { throw std::runtime_error(errorMessage); }

    [[nodiscard]] const std::vector<std::string>& GetSpecies() const override { throw std::runtime_error(errorMessage); }

    [[nodiscard]] const std::vector<std::string>& GetReferenceSpecies() const { throw std::runtime_error(errorMessage); }

    [[nodiscard]] const std::vector<std::string>& GetExtraVariables() const override { throw std::runtime_error(errorMessage); }

    std::shared_ptr<SourceCalculator> CreateSourceCalculator(const std::vector<domain::Field>& fields, const solver::Range& cellRange) override { throw std::runtime_error(errorMessage); }

    void View(std::ostream& stream) const override { throw std::runtime_error(errorMessage); }

    [[nodiscard]] eos::ThermodynamicFunction GetThermodynamicFunction(eos::ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override {
        throw std::runtime_error(errorMessage);
    }

    [[nodiscard]] eos::ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override {
        throw std::runtime_error(errorMessage);
    }

    [[nodiscard]] eos::FieldFunction GetFieldFunctionFunction(const std::string& field, eos::ThermodynamicProperty property1, eos::ThermodynamicProperty property2) const override {
        throw std::runtime_error(errorMessage);
    }
};
#endif
}  // namespace ablate::eos
#endif  // ABLATELIBRARY_CHEMTABMODEL_HPP
