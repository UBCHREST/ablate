#ifndef ABLATELIBRARY_CHEMTAB_HPP
#define ABLATELIBRARY_CHEMTAB_HPP

#include <petscmat.h>
#include <filesystem>
#include <istream>
#include "chemistryModel.hpp"
#include "eos/tChem.hpp"
#ifdef WITH_TENSORFLOW
#include <tensorflow/c/c_api.h>
#include "utilities/vectorUtilities.hpp"
#endif

namespace ablate::eos {

#ifdef WITH_TENSORFLOW
class ChemTab : public ChemistryModel, public std::enable_shared_from_this<ChemTab> {
   private:
    //! use the reference eos to compute properties from the decoded progressVariables to yi
    std::shared_ptr<ablate::eos::TChem> referenceEOS;

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
     * The source calculator is used to do batch processing for chemistry model.  This is a bad implementation
     * that calls each node one at a time.
     */
    class ChemTabSourceCalculator : public ChemistryModel::SourceCalculator {
       private:
        //! Store the offset for each of the required variables
        const PetscInt densityOffset;
        const PetscInt densityEnergyOffset;
        const PetscInt densityProgressVariableOffset;
        //! hold a pointer to the chemTabModel to compute the source terms
        const std::shared_ptr<ChemTab> chemTabModel;

       public:
        ChemTabSourceCalculator(PetscInt densityOffset, PetscInt densityEnergyOffset, PetscInt densityProgressVariableOffset, std::shared_ptr<ChemTab> chemTabModel);
        /**
         * There is no need to precompute source for the chemtab model
         */
        void ComputeSource(const solver::Range& cellRange, PetscReal time, PetscReal dt, Vec solution) override{};

        /**
         * Computes and adds the source to the supplied vector
         */
        void AddSource(const solver::Range& cellRange, Vec locSolution, Vec locSource) override;
    };

    /**
     * Struct for the thermodynamic function context
     */
    struct ThermodynamicFunctionContext {
        // memory access locations for fields
        std::size_t numberSpecies;
        std::size_t numberProgressVariables;
        std::size_t densityOffset;
        std::size_t progressOffset;

        // store a scratch variable to compute yi
        std::vector<PetscReal> yiScratch;

        // Hold the context for the baseline tChem function
        ablate::eos::TChem::ThermodynamicMassFractionFunction tChemFunction;

        // inverse function/  This does not hold the pointer, but it is held by chemTab;
        PetscReal** iWmat;
    };

    /**
     * Struct for the thermodynamic function context
     */
    struct ThermodynamicTemperatureFunctionContext {
        // memory access locations for fields
        std::size_t numberSpecies;
        std::size_t numberProgressVariables;
        std::size_t densityOffset;
        std::size_t progressOffset;

        // store a scratch variable to compute yi
        std::vector<PetscReal> yiScratch;

        // Hold the context for the baseline tChem function
        ablate::eos::TChem::ThermodynamicTemperatureMassFractionFunction tChemFunction;

        // inverse function/  This does not hold the pointer, but it is held by chemTab;
        PetscReal** iWmat;
    };

    /**
     * static call to compute yi and call baseline tChem function
     * @param conserved
     * @param property
     * @param ctx
     * @return
     */
    static PetscErrorCode ChemTabThermodynamicFunction(const PetscReal conserved[], PetscReal* property, void* ctx);

    /**
     * static call to compute yi and call baseline tChem function
     * @param conserved
     * @param property
     * @param ctx
     * @return
     */
    static PetscErrorCode ChemTabThermodynamicTemperatureFunction(const PetscReal conserved[], PetscReal T, PetscReal* property, void* ctx);

    /**
     * helper function to compute the mass fractions = from the mass fractions progress variables
     * @param massFractions
     * @param massFractionsSize
     * @param progressVariables
     * @param progressVariablesSize
     * @param density allows for this function to be used with density*progress variables
     */
    static void ComputeMassFractions(std::size_t numSpecies, std::size_t numProgressVariables, PetscReal** iWmat, const PetscReal* progressVariables, PetscReal* massFractions,
                                     PetscReal density = 1.0);

   public:
    explicit ChemTab(std::filesystem::path path);
    ~ChemTab() override;

    /**
     * As far as other parts of the code is concerned the chemTabEos does not expect species to be transported
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetSpeciesVariables() const override { return ablate::utilities::VectorUtilities::Empty<std::string>; }

    /**
     * List of species used for the field function initialization.
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetSpecies() const override { return referenceEOS->GetSpecies(); }

    /**
     * As far as other parts of the code is concerned the chemTabEos does not expect species
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetProgressVariables() const override { return progressVariablesNames; }

    /**
     * Single function to compute the source terms for a single point
     * @param fields
     * @param conserved
     * @param source
     */
    void ChemistrySource(PetscReal density, const PetscReal densityProgressVariable[], PetscReal* densityEnergySource, PetscReal* progressVariableSource) const;

    /**
     * Single function to produce ChemistryFunction calculator based upon the available fields and sources.
     * @param fields in the conserved/source arrays
     * @param property
     * @param fields
     * @return
     */
    std::shared_ptr<SourceCalculator> CreateSourceCalculator(const std::vector<domain::Field>& fields, const solver::Range& cellRange) override;

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
     * @param density allows for this function to be used with density*progress variables
     *
     */
    void ComputeMassFractions(const PetscReal* progressVariables, std::size_t progressVariablesSize, PetscReal* massFractions, std::size_t massFractionsSize, PetscReal density = 1.0) const;

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
    [[nodiscard]] eos::ThermodynamicFunction GetThermodynamicFunction(eos::ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] eos::ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override;

    /**
     * Single function to produce fieldFunction function for any two properties, velocity, and species mass fractions.  These calls can be slower and should be used for init/output only
     * @param field
     * @param property1
     * @param property2
     */
    [[nodiscard]] eos::FieldFunction GetFieldFunctionFunction(const std::string& field, eos::ThermodynamicProperty property1, eos::ThermodynamicProperty property2) const override;
};

#else
class ChemTab : public ChemistryModel {
   public:
    static inline const std::string errorMessage = "Using the ChemTab requires Tensorflow to be compile with ABLATE.";
    ChemTab(std::filesystem::path path) : ChemistryModel("ablate::chemistry::ChemTabModel") { throw std::runtime_error(errorMessage); }

    [[nodiscard]] const std::vector<std::string>& GetSpeciesVariables() const override { throw std::runtime_error(errorMessage); }

    [[nodiscard]] const std::vector<std::string>& GetSpecies() const override { throw std::runtime_error(errorMessage); }

    [[nodiscard]] const std::vector<std::string>& GetProgressVariables() const override { throw std::runtime_error(errorMessage); }

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

    void ChemistrySource(PetscReal density, const PetscReal densityProgressVariable[], PetscReal* densityEnergySource, PetscReal* progressVariableSource) const {
        throw std::runtime_error(errorMessage);
    }

    void ComputeProgressVariables(const PetscReal* massFractions, std::size_t massFractionsSize, PetscReal* progressVariables, std::size_t progressVariablesSize) const {
        throw std::runtime_error(errorMessage);
    }

    void ComputeMassFractions(const PetscReal* progressVariables, std::size_t progressVariablesSize, PetscReal* massFractions, std::size_t massFractionsSize, PetscReal density = 1.0) {
        throw std::runtime_error(errorMessage);
    }
};
#endif
}  // namespace ablate::eos
#endif  // ABLATELIBRARY_CHEMTAB_HPP
