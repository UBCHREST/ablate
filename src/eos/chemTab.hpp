#ifndef ABLATELIBRARY_CHEMTAB_HPP
#define ABLATELIBRARY_CHEMTAB_HPP

#include <petscmat.h>
#include <filesystem>
#include <istream>
#include "chemistryModel.hpp"
#include "eos/tChem.hpp"
#ifdef WITH_TENSORFLOW
#include <tensorflow/c/c_api.h>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/vectorUtilities.hpp"
#endif

namespace ablate::eos {

#ifdef WITH_TENSORFLOW
class ChemTab : public ChemistryModel, public std::enable_shared_from_this<ChemTab>, public utilities::Loggable<ChemTab> {
   public:
    inline const static std::string DENSITY_YI_DECODE_FIELD = "DENSITY_YI_DECODE";

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

    // Store any initializers specified by the metadata
    std::map<std::string, std::map<std::string, double>> initializers;

    /**
     * private implementations of support functions
     */
    void ExtractMetaData(std::istream& inputStream);
    static void LoadBasisVectors(std::istream& inputStream, std::size_t columns, PetscReal** W);

    /**
     * Private function to compute predictedSourceEnergy, progressVariableSource, and massFractions
     * @param density, the density is used to scale both the progress variable and resulting densityMassFractions
     * @param densityProgressVariables
     * @param densityEnergySource , if null, wont' be set
     * @param densityProgressVariableSource , if null, won't be set
     * @param densityMassFractions , if null, won't be set
     */
    void ChemTabModelComputeFunction(PetscReal density, const PetscReal densityProgressVariables[],
                                     PetscReal* densityEnergySource, PetscReal* densityProgressVariableSource,
                                     PetscReal* densityMassFractions) const;
    void ChemTabModelComputeFunction(const PetscReal density[], const PetscReal*const*const densityProgressVariables,
                                     PetscReal** densityEnergySource, PetscReal** densityProgressVariableSource,
                                     PetscReal** densityMassFractions, size_t batch_size) const;
    // Batched Version of above

    //! Tell the compressible flow fields what tags to use with this field
    [[nodiscard]] std::vector<std::string> GetFieldTags() const override { return std::vector<std::string>{ablate::finiteVolume::CompressibleFlowFields::MinusOneToOneRange}; }

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
        void ComputeSource(const ablate::domain::Range& cellRange, PetscReal time, PetscReal dt, Vec solution) override{};

        /**
         * Computes and adds the source to the supplied vector
         */
        void AddSource(const ablate::domain::Range& cellRange, Vec locSolution, Vec locSource) override;
    };

    /**
     * helper function to compute the progress variables from the mass fractions
     * @param massFractions
     * @param massFractionsSize
     * @param progressVariables
     * @param progressVariablesSize
     */
    void ComputeProgressVariables(const PetscReal* massFractions, PetscReal* progressVariables) const;
    void ComputeProgressVariables(const PetscReal*const* massFractions, PetscReal*const* progressVariables, size_t n) const;
    // Batched version of above

    /**
     * private function to compute the mass fractions assuming euler[0] and densityProgressVariable[1] and densityYi[2] is provided
     * @param time
     * @param dim
     * @param cellGeom
     * @param uOff
     * @param u
     * @param ctx
     * @return
     */
    static PetscErrorCode ComputeMassFractions(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], PetscScalar* u, void* ctx);

   public:
    explicit ChemTab(const std::filesystem::path& path);
    ~ChemTab() override;

    /**
     * As far as other parts of the code is concerned the chemTabEos does not expect species to be transported
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetSpeciesVariables() const override { return ablate::utilities::VectorUtilities::Empty<std::string>; }

    /**
     * Function used to get the species used in the chem tab model
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetSpeciesNames() const { return speciesNames; }

    /**
     * List of species used for the field function initialization.
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetFieldFunctionProperties() const override { return progressVariablesNames; }

    /**
     * As far as other parts of the code is concerned the chemTabEos does not expect species
     * @return
     */
    [[nodiscard]] const std::vector<std::string>& GetProgressVariables() const override { return progressVariablesNames; }

    /**
     * As far as other parts of the code is concerned the chemTabEos does not expect species
     * @return
     */
    std::vector<std::shared_ptr<domain::FieldDescriptor>> GetAdditionalFields() const override;

    /**
     * Single function to compute the source terms for a single point
     * @param fields
     * @param conserved
     * @param source
     */
    void ChemistrySource(const PetscReal density, const PetscReal densityProgressVariable[],
                         PetscReal* densityEnergySource, PetscReal* densityProgressVariableSource) const;
    void ChemistrySource(const PetscReal *const density, const PetscReal *const*const densityProgressVariable,
                         PetscReal** densityEnergySource, PetscReal** progressVariableSource, size_t n) const;
    // Batched version of above


    /**
     * Single function to produce ChemistryFunction calculator based upon the available fields and sources.
     * @param fields in the conserved/source arrays
     * @param property
     * @param fields
     * @return
     */
    std::shared_ptr<SourceCalculator> CreateSourceCalculator(const std::vector<domain::Field>& fields, const ablate::domain::Range& cellRange) override;

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
    [[nodiscard]] eos::EOSFunction GetFieldFunctionFunction(const std::string& field, eos::ThermodynamicProperty property1, eos::ThermodynamicProperty property2,
                                                            std::vector<std::string> otherProperties) const override;

    /**
     * public function to compute the progress variables from the mass fractions
     * @param massFractions
     * @param massFractionsSize
     * @param progressVariables
     * @param progressVariablesSize
     */
    void ComputeProgressVariables(const std::vector<PetscReal>& massFractions, std::vector<PetscReal>& progressVariables) const;

    /**
     * public function to compute the mass fractions = from the mass fractions progress variables
     * @param massFractions
     * @param massFractionsSize
     * @param progressVariables
     * @param progressVariablesSize
     * @param density allows for this function to be used with density*progress variables
     *
     */
    void ComputeMassFractions(std::vector<PetscReal>& progressVariables, std::vector<PetscReal>& massFractions, PetscReal density = 1.0) const;

//    /**
//     * helper function to compute the mass fractions = from the mass fractions progress variables
//     * @param progressVariables is density*progress
//     * @param massFractions
//     * @param density allows for this function to be used with density*progress variables
//     */
//    void ComputeMassFractions(PetscReal* progressVariables, PetscReal* massFractions, PetscReal density = 1.0) const;

    /**
     * helper function to compute the mass fractions = from the mass fractions progress variables
     * @param progressVariables is density*progress
     * @param massFractions
     * @param density allows for this function to be used with density*progress variables
     */
    void ComputeMassFractions(const PetscReal* densityProgressVariables, PetscReal* densityMassFractions, const PetscReal density = 1.0) const;
    void ComputeMassFractions(const PetscReal*const* densityProgressVariables, PetscReal** densityMassFractions, const PetscReal density[], size_t n) const;
    // Batched version of above

    /**
     * Computes the progress variables for a given initializer
     * @param name
     * @param progressVariables
     */
    void GetInitializerProgressVariables(const std::string& name, std::vector<PetscReal>& progressVariables) const;

    /**
     * Return a function to update the densityYi based upon the current progress variable
     * @return
     */
    [[nodiscard]] std::vector<std::tuple<ablate::solver::CellSolver::SolutionFieldUpdateFunction, void*, std::vector<std::string>>> GetSolutionFieldUpdates() override;

    void extractModelOutputsAtPoint(const PetscReal density, PetscReal *densityEnergySource,
                                    PetscReal *densityProgressVariableSource, PetscReal *densityMassFractions,
                                    const std::array<TF_Tensor *, 2> &outputValues, size_t id=0) const;
    };

#else
class ChemTab : public ChemistryModel {
   public:
    inline const static std::string DENSITY_YI_DECODE_FIELD = "DENSITY_YI_DECODE";
    static inline const std::string errorMessage = "Using the ChemTab requires Tensorflow to be compile with ABLATE.";
    ChemTab(std::filesystem::path path) : ChemistryModel("ablate::chemistry::ChemTabModel") { throw std::runtime_error(errorMessage); }

    [[nodiscard]] const std::vector<std::string>& GetSpeciesVariables() const override { throw std::runtime_error(errorMessage); }

    [[nodiscard]] const std::vector<std::string>& GetSpeciesNames() const { throw std::runtime_error(errorMessage); }

    [[nodiscard]] const std::vector<std::string>& GetFieldFunctionProperties() const override { throw std::runtime_error(errorMessage); }

    [[nodiscard]] const std::vector<std::string>& GetProgressVariables() const override { throw std::runtime_error(errorMessage); }

    [[nodiscard]] std::vector<std::string> GetFieldTags() const override { throw std::runtime_error(errorMessage); }

    std::shared_ptr<SourceCalculator> CreateSourceCalculator(const std::vector<domain::Field>& fields, const ablate::domain::Range& cellRange) override { throw std::runtime_error(errorMessage); }

    void View(std::ostream& stream) const override { throw std::runtime_error(errorMessage); }

    [[nodiscard]] eos::ThermodynamicFunction GetThermodynamicFunction(eos::ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override {
        throw std::runtime_error(errorMessage);
    }

    [[nodiscard]] eos::ThermodynamicTemperatureFunction GetThermodynamicTemperatureFunction(eos::ThermodynamicProperty property, const std::vector<domain::Field>& fields) const override {
        throw std::runtime_error(errorMessage);
    }

    [[nodiscard]] eos::EOSFunction GetFieldFunctionFunction(const std::string& field, eos::ThermodynamicProperty property1, eos::ThermodynamicProperty property2,
                                                            std::vector<std::string> otherProperties) const override {
        throw std::runtime_error(errorMessage);
    }

    [[nodiscard]] ThermodynamicTemperatureMassFractionFunction GetThermodynamicTemperatureMassFractionFunction(ThermodynamicProperty property,
                                                                                                               const std::vector<domain::Field>& fields) const override {
        throw std::runtime_error(errorMessage);
    }
    std::map<std::string, double> GetSpeciesMolecularMass() const override { throw std::runtime_error(errorMessage); }
    std::map<std::string, std::map<std::string, int>> GetSpeciesElementalInformation() const override { throw std::runtime_error(errorMessage); }
    std::map<std::string, double> GetElementInformation() const override { throw std::runtime_error(errorMessage); }

    void ChemistrySource(PetscReal density, const PetscReal densityProgressVariable[], PetscReal* densityEnergySource, PetscReal* progressVariableSource) const {
        throw std::runtime_error(errorMessage);
    }

    void ComputeProgressVariables(const std::vector<PetscReal>& massFractions, std::vector<PetscReal>& progressVariables) const { throw std::runtime_error(errorMessage); }

    void ComputeMassFractions(const std::vector<PetscReal>& progressVariables, std::vector<PetscReal>& massFractions, PetscReal density = 1.0) const { throw std::runtime_error(errorMessage); }

    void ComputeMassFractions(const PetscReal* progressVariables, PetscReal* massFractions, PetscReal density = 1.0) const { throw std::runtime_error(errorMessage); }

    void GetInitializerProgressVariables(const std::string& name, std::vector<PetscReal>& progressVariables) const { throw std::runtime_error(errorMessage); }

    [[nodiscard]] std::vector<std::tuple<ablate::solver::CellSolver::SolutionFieldUpdateFunction, void*, std::vector<std::string>>> GetSolutionFieldUpdates() override {
        throw std::runtime_error(errorMessage);
    }
};
#endif
}  // namespace ablate::eos
#endif  // ABLATELIBRARY_CHEMTAB_HPP
