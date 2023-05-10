#include "chemTab.hpp"

#include <eos/tChem.hpp>
#include <fstream>

#ifdef WITH_TENSORFLOW
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include "finiteVolume/compressibleFlowFields.hpp"

static void NoOpDeallocator(void *, size_t, void *) {}

ablate::eos::ChemTab::ChemTab(const std::filesystem::path &path) : ChemistryModel("ablate::chemistry::ChemTab") {
    const char *tags = "serve";  // default model serving tag; can change in future
    int ntags = 1;

    // check to make sure the file
    if (!exists(path)) {
        throw std::runtime_error("Cannot locate ChemTab Folder " + path.string());
    }

    // open the metadata yaml file
    auto metadata = YAML::LoadFile(path / "metadata.yaml");
    const std::string rpath = path / metadata["rpath"].as<std::string>();
    const std::string wpath = path / metadata["wpath"].as<std::string>();

    // Check for missing files
    if (!std::filesystem::exists(rpath)) {
        throw std::runtime_error(
            "The 'regressor' file cannot be located in the "
            "specified ChemTab Folder " +
            path.string());
    }
    if (!std::filesystem::exists(wpath)) {
        throw std::runtime_error(
            "The 'weights.csv' file cannot be located in the "
            "specified ChemTab Folder " +
            path.string());
    }

    // Load in any initializers from the metadata
    for (const auto &node : metadata["initializers"]) {
        initializers[node.first.as<std::string>()] = node.second["mass_fractions"].as<std::map<std::string, double>>();
    }

    // Load the source energy predictor model first
    graph = TF_NewGraph();
    status = TF_NewStatus();
    sessionOpts = TF_NewSessionOptions();
    runOpts = nullptr;
    session = TF_LoadSessionFromSavedModel(sessionOpts, runOpts, rpath.c_str(), &tags, ntags, graph, nullptr, status);

    std::fstream inputFileStream;
    // load the meta data from the weights.csv file
    inputFileStream.open(wpath.c_str(), std::ios::in);
    ExtractMetaData(inputFileStream);
    inputFileStream.close();

    // load the basis vectors from the weights.csv
    // first allocate memory for both weight matrices
    Wmat = (PetscReal **)malloc(speciesNames.size() * sizeof(PetscReal *));
    for (std::size_t i = 0; i < speciesNames.size(); i++) {
        Wmat[i] = (PetscReal *)malloc(progressVariablesNames.size() * sizeof(PetscReal));
    }
    inputFileStream.open(wpath.c_str(), std::ios::in);
    LoadBasisVectors(inputFileStream, progressVariablesNames.size(), Wmat);
    inputFileStream.close();

    // create a reference equation of state given the mechanism provided in the metedata file
    const std::string mechanismPath = path / metadata["mechanism"].as<std::string>();
    referenceEOS = std::make_shared<ablate::eos::TChem>(mechanismPath);

    // make sure that the species list is the same
    auto &referenceEOSSpecies = referenceEOS->GetSpeciesVariables();
    if (referenceEOSSpecies.size() != speciesNames.size()) {
        throw std::invalid_argument("The ReferenceEOS species and chemTab species are expected to be the same.");
    }
    for (std::size_t s = 0; s < speciesNames.size(); s++) {
        if (speciesNames[s] != referenceEOSSpecies[s]) {
            throw std::invalid_argument("The ReferenceEOS species and chemTab species are expected to be the same.");
        }
    }
}

ablate::eos::ChemTab::~ChemTab() {
    TF_DeleteGraph(graph);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(sessionOpts);
    TF_DeleteStatus(status);
    free(sourceEnergyScaler);
    for (std::size_t i = 0; i < speciesNames.size(); i++) free(Wmat[i]);

    free(Wmat);
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

void ablate::eos::ChemTab::ExtractMetaData(std::istream &inputStream) {
    // determine the headers from the first row
    std::string line;
    std::getline(inputStream, line);
    int i = 0;

    // get the progress variable names (skipping the first one)
    std::stringstream headerStream(line);
    while (headerStream.good()) {
        std::string headerColumn;
        getline(headerStream, headerColumn, ',');  // delimited by comma
        if (i > 0) {
            trim(headerColumn);
            progressVariablesNames.push_back(headerColumn);
        }
        i++;
    }
    // parse each line after header, first entry in each line is the name of the
    // corresponding species
    while (std::getline(inputStream, line)) {
        std::istringstream lineStream(line);
        std::string sName;
        getline(lineStream, sName, ',');
        speciesNames.push_back(sName);
    }
}

void ablate::eos::ChemTab::LoadBasisVectors(std::istream &inputStream, std::size_t cols, double **W) {
    std::string line, sName;
    // skip first row
    std::getline(inputStream, line);
    // parse each line after header, first entry in each line is the name of the
    // corresponding species, followed by the values
    int i = 0;
    while (std::getline(inputStream, line)) {
        std::istringstream lineStream(line);
        // skip the first entry
        getline(lineStream, sName, ',');
        for (std::size_t j = 0; j < cols; j++) {
            std::string val;
            getline(lineStream, val, ',');  // delimited by comma
            W[i][j] = std::stod(val);
        }
        i++;
    }
}

// avoids freeing null pointers
#define safe_free(ptr) \
    if (ptr != NULL) free(ptr)

void ablate::eos::ChemTab::ChemTabModelComputeFunction(PetscReal density, const PetscReal densityProgressVariable[], PetscReal *predictedSourceEnergy, PetscReal *progressVariableSource,
                                                       PetscReal *massFractions) const {
    //********* Get Input tensor
    const std::size_t numInputs = 1;

    TF_Output t0 = {TF_GraphOperationByName(graph, "serving_default_input_1"), 0};

    if (t0.oper == nullptr) throw std::runtime_error("ERROR: Failed TF_GraphOperationByName serving_default_input_1");
    std::array<TF_Output, numInputs> input = {t0};

    //********* Get Output tensor
    const std::size_t numOutputs = 2;

    TF_Output t_sourceenergy = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};
    TF_Output t_sourceterms = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 1};

    if (t_sourceenergy.oper == nullptr) throw std::runtime_error("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall:0");
    if (t_sourceterms.oper == nullptr) throw std::runtime_error("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall:1");
    std::array<TF_Output, numOutputs> output = {t_sourceenergy, t_sourceterms};

    //********* Allocate data for inputs & outputs
    std::array<TF_Tensor*, numInputs> inputValues = {nullptr};
    std::array<TF_Tensor*, numOutputs> outputValues= {nullptr, nullptr};

    std::size_t ndims = 2;

    // according to Varun this should work for including Zmix
    auto ninputs = progressVariablesNames.size();
    int64_t dims[] = {1, (int)ninputs};
    float data[ninputs];

    for (std::size_t i = 0; i < ninputs; i++) {
        data[i] = (float)(densityProgressVariable[i] / density);
    }

    std::size_t ndata = ninputs * sizeof(float);
    TF_Tensor *input_tensor = TF_NewTensor(TF_FLOAT, dims, (int)ndims, data, ndata, &NoOpDeallocator, nullptr);
    if (input_tensor == nullptr) throw std::runtime_error("ERROR: Failed TF_NewTensor");

    inputValues[0] = input_tensor;

    TF_SessionRun(session, nullptr, input.data(), inputValues.data(), (int)numInputs, output.data(), outputValues.data(), (int)numOutputs, nullptr, 0, nullptr, status);
    if (TF_GetCode(status) != TF_OK) throw std::runtime_error(TF_Message(status));
    //********** Extract source predictions

    // store physical variables (e.g. souener & mass fractions)
    float *outputArray;  // Dwyer: as counter intuitive as it may be static dependents come second, it did pass its tests!
    outputArray = (float *)TF_TensorData(outputValues[1]);
    auto p = (PetscReal)outputArray[0];
    if (predictedSourceEnergy != nullptr) *predictedSourceEnergy += p;

    // store inverted mass fractions
    if (massFractions) {
        for (size_t i = 0; i < speciesNames.size(); i++) {
            massFractions[i] = (PetscReal)outputArray[i + 1];  // i+1 b/c i==0 is souener!
        }
    }

    // store CPV sources
    outputArray = (float *)TF_TensorData(outputValues[0]);
    if (progressVariableSource != nullptr) {
        progressVariableSource[0] = 0;  // Zmix source is always 0!

        // -1 b/c we don't want to go out of bounds with the +1 below, also int is to prevent integer overflow
        for (size_t i = 0; i < (progressVariablesNames.size() - 1); ++i) {
            progressVariableSource[i + 1] = (PetscReal)outputArray[i];  // +1 b/c we are manually filling in Zmix source value (to 0)
        }
    }

    // free allocated vectors
    for (auto& t : outputValues) {
        TF_DeleteTensor(t);
    }
    for (auto& t : inputValues) {
        TF_DeleteTensor(t);
    }
}

void ablate::eos::ChemTab::ComputeMassFractions(const PetscReal *progressVariables, PetscReal *massFractions, PetscReal density) const {
    // call model using generalized invocation method (usable for inversion & source computation)
    ChemTabModelComputeFunction(density, progressVariables, nullptr, nullptr, massFractions);
}

void ablate::eos::ChemTab::ComputeMassFractions(const std::vector<PetscReal> &progressVariables, std::vector<PetscReal> &massFractions, PetscReal density) const {
    if (progressVariables.size() != progressVariablesNames.size()) {
        throw std::invalid_argument("The Progress variable size is expected to be " + std::to_string(progressVariablesNames.size()));
    }
    if (massFractions.size() != speciesNames.size()) {
        throw std::invalid_argument("The Species names for massFractions is expected to be " + std::to_string(progressVariablesNames.size()));
    }
    ComputeMassFractions(progressVariables.data(), massFractions.data(), density);
}

void ablate::eos::ChemTab::ComputeProgressVariables(const std::vector<PetscReal> &massFractions, std::vector<PetscReal> &progressVariables) const {
    if (progressVariables.size() != progressVariablesNames.size()) {
        throw std::invalid_argument("The Progress variable size is expected to be " + std::to_string(progressVariablesNames.size()));
    }
    if (massFractions.size() != speciesNames.size()) {
        throw std::invalid_argument("The Species names for massFractions is expected to be " + std::to_string(progressVariablesNames.size()));
    }
    ComputeProgressVariables(massFractions.data(), progressVariables.data());
}

void ablate::eos::ChemTab::ComputeProgressVariables(const PetscReal *massFractions, PetscReal *progressVariables) const {
    // c = W'y
    for (size_t i = 0; i < progressVariablesNames.size(); i++) {
        PetscReal v = 0;
        for (size_t j = 0; j < speciesNames.size(); j++) {
            v += Wmat[j][i] * massFractions[j];
        }
        progressVariables[i] = v;
    }
}

void ablate::eos::ChemTab::ChemistrySource(PetscReal density, const PetscReal densityProgressVariable[], PetscReal *densityEnergySource, PetscReal *progressVariableSource) const {
    // call model using generalized invocation method (usable for inversion & source computation)
    ChemTabModelComputeFunction(density, densityProgressVariable, densityEnergySource, progressVariableSource, nullptr);
}

void ablate::eos::ChemTab::View(std::ostream &stream) const { stream << "EOS: " << type << std::endl; }

std::shared_ptr<ablate::eos::ChemistryModel::SourceCalculator> ablate::eos::ChemTab::CreateSourceCalculator(const std::vector<domain::Field> &fields, const ablate::domain::Range &cellRange) {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::chemistry::ChemTabModel requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    auto densityProgressField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD; });
    if (densityProgressField == fields.end()) {
        throw std::invalid_argument("The ablate::chemistry::ChemTabModel requires the ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD Field");
    }

    return std::make_shared<ChemTabSourceCalculator>(eulerField->offset + ablate::finiteVolume::CompressibleFlowFields::RHO,
                                                     eulerField->offset + ablate::finiteVolume::CompressibleFlowFields::RHOE,
                                                     densityProgressField->offset,
                                                     shared_from_this());
}
PetscErrorCode ablate::eos::ChemTab::ChemTabThermodynamicFunction(const PetscReal *conserved, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (ThermodynamicFunctionContext *)ctx;

    // fill the mass fractions
    functionContext->chemTab->ComputeMassFractions(conserved + functionContext->progressOffset, functionContext->yiScratch.data(), conserved[functionContext->densityOffset]);

    // call the tChem function
    PetscCall(functionContext->tChemFunction.function(conserved, functionContext->yiScratch.data(), property, functionContext->tChemFunction.context.get()));

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::ChemTab::ChemTabThermodynamicTemperatureFunction(const PetscReal *conserved, PetscReal T, PetscReal *property, void *ctx) {
    PetscFunctionBeginUser;
    auto functionContext = (ThermodynamicTemperatureFunctionContext *)ctx;

    // fill the mass fractions
    functionContext->chemTab->ComputeMassFractions(conserved + functionContext->progressOffset, functionContext->yiScratch.data(), conserved[functionContext->densityOffset]);

    // call the tChem function
    PetscCall(functionContext->tChemFunction.function(conserved, functionContext->yiScratch.data(), T, property, functionContext->tChemFunction.context.get()));

    PetscFunctionReturn(0);
}

ablate::eos::ThermodynamicFunction ablate::eos::ChemTab::GetThermodynamicFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::TChem requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    auto densityProgressField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD; });
    if (densityProgressField == fields.end()) {
        throw std::invalid_argument("The ablate::chemistry::ChemTabModel requires the ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD Field");
    }
    auto tChemThermodynamicFunction = referenceEOS->GetThermodynamicMassFractionFunction(property, fields);
    return ThermodynamicFunction{
        .function = ChemTabThermodynamicFunction,
        .context = std::make_shared<ThermodynamicFunctionContext>(ThermodynamicFunctionContext{.densityOffset = eulerField->offset + (std::size_t)ablate::finiteVolume::CompressibleFlowFields::RHO,
                                                                                               .progressOffset = (std::size_t)densityProgressField->offset,
                                                                                               .yiScratch = std::vector<PetscReal>(speciesNames.size()),
                                                                                               .tChemFunction = tChemThermodynamicFunction,
                                                                                               .chemTab = shared_from_this()}),
        .propertySize = tChemThermodynamicFunction.propertySize};
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::ChemTab::GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    // Look for the euler field
    auto eulerField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD; });
    if (eulerField == fields.end()) {
        throw std::invalid_argument("The ablate::eos::TChem requires the ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD Field");
    }

    auto densityProgressField = std::find_if(fields.begin(), fields.end(), [](const auto &field) { return field.name == ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD; });
    if (densityProgressField == fields.end()) {
        throw std::invalid_argument("The ablate::chemistry::ChemTabModel requires the ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD Field");
    }

    auto tChemThermodynamicFunction = referenceEOS->GetThermodynamicTemperatureMassFractionFunction(property, fields);
    return ThermodynamicTemperatureFunction{.function = ChemTabThermodynamicTemperatureFunction,
                                            .context = std::make_shared<ThermodynamicTemperatureFunctionContext>(
                                                ThermodynamicTemperatureFunctionContext{.densityOffset = eulerField->offset + (std::size_t)ablate::finiteVolume::CompressibleFlowFields::RHO,
                                                                                        .progressOffset = (std::size_t)densityProgressField->offset,
                                                                                        .yiScratch = std::vector<PetscReal>(speciesNames.size()),
                                                                                        .tChemFunction = tChemThermodynamicFunction,
                                                                                        .chemTab = shared_from_this()}),
                                            .propertySize = tChemThermodynamicFunction.propertySize};
}

ablate::eos::EOSFunction ablate::eos::ChemTab::GetFieldFunctionFunction(const std::string &field, eos::ThermodynamicProperty property1, eos::ThermodynamicProperty property2,
                                                                        std::vector<std::string> otherProperties) const {
    if (finiteVolume::CompressibleFlowFields::EULER_FIELD == field && otherProperties == std::vector<std::string>{YI}) {
        return referenceEOS->GetFieldFunctionFunction(field, property1, property2, otherProperties);
    } else if (finiteVolume::CompressibleFlowFields::EULER_FIELD == field && otherProperties == std::vector<std::string>{PROGRESS}) {
        auto eulerFunction = referenceEOS->GetFieldFunctionFunction(finiteVolume::CompressibleFlowFields::EULER_FIELD, property1, property2, {YI});

        return [=](PetscReal property1, PetscReal property2, PetscInt dim, const PetscReal velocity[], const PetscReal progress[], PetscReal conserved[]) {
            // Compute the mass fractions from progress
            std::vector<PetscReal> yi(speciesNames.size());

            // compute the progress variables and put into conserved for now
            ComputeMassFractions(progress, yi.data());

            // Scale the progress variables by density
            eulerFunction(property1, property2, dim, velocity, yi.data(), conserved);
        };
    } else if (finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD == field && otherProperties == std::vector<std::string>{YI}) {
        // get the euler field because we need density
        auto eulerFunction = referenceEOS->GetFieldFunctionFunction(finiteVolume::CompressibleFlowFields::EULER_FIELD, property1, property2, otherProperties);

        return [=](PetscReal property1, PetscReal property2, PetscInt dim, const PetscReal velocity[], const PetscReal yi[], PetscReal conserved[]) {
            // Compute euler
            PetscReal euler[ablate::finiteVolume::CompressibleFlowFields::RHOW + 1];  // Max size for euler
            eulerFunction(property1, property2, dim, velocity, yi, euler);

            // compute the progress variables and put into conserved for now
            ComputeProgressVariables(yi, conserved);

            // Scale the progress variables by density
            for (std::size_t p = 0; p < progressVariablesNames.size(); p++) {
                conserved[p] *= euler[ablate::finiteVolume::CompressibleFlowFields::RHO];
            }
        };
    } else if (finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD == field && otherProperties == std::vector<std::string>{PROGRESS}) {
        // get the euler field because we need density
        auto eulerFunction = referenceEOS->GetFieldFunctionFunction(finiteVolume::CompressibleFlowFields::EULER_FIELD, property1, property2, {YI});

        return [=](PetscReal property1, PetscReal property2, PetscInt dim, const PetscReal velocity[], const PetscReal progress[], PetscReal conserved[]) {
            // Compute the mass fractions from progress
            PetscReal yi[speciesNames.size()];

            // compute the progress variables and put into conserved for now
            ComputeMassFractions(progress, yi);

            // Compute euler
            PetscReal euler[ablate::finiteVolume::CompressibleFlowFields::RHOW + 1];  // Max size for euler
            eulerFunction(property1, property2, dim, velocity, yi, euler);

            // Scale the progress variables by density
            for (std::size_t p = 0; p < progressVariablesNames.size(); p++) {
                conserved[p] = euler[ablate::finiteVolume::CompressibleFlowFields::RHO] * progress[p];
            }
        };
    } else {
        throw std::invalid_argument("Unknown field type " + field + " and otherProperties " + ablate::utilities::VectorUtilities::Concatenate(otherProperties) + " for ablate::eos::ChemTab.");
    }
}
void ablate::eos::ChemTab::GetInitializerProgressVariables(const std::string &name, std::vector<PetscReal> &progressVariables) const {
    // Fill the mass fractions based upon the initializer
    if (!initializers.count(name)) {
        throw std::invalid_argument("The initializers " + name + " cannot be found.");
    }

    std::vector<PetscReal> yiScratch(speciesNames.size(), 0.0);
    for (const auto &[species, value] : initializers.at(name)) {
        auto loc = std::find(speciesNames.begin(), speciesNames.end(), species);
        if (loc == speciesNames.end()) {
            throw std::invalid_argument("Unable to locate species " + species);
        }
        yiScratch[std::distance(speciesNames.begin(), loc)] = value;
    }

    // Compute the progress variables
    progressVariables.resize(progressVariablesNames.size());
    ComputeProgressVariables(yiScratch, progressVariables);
}

ablate::eos::ChemTab::ChemTabSourceCalculator::ChemTabSourceCalculator(PetscInt densityOffset, PetscInt densityEnergyOffset, PetscInt densityProgressVariableOffset,
                                                                       std::shared_ptr<ChemTab> chemTabModel)
    : densityOffset(densityOffset), densityEnergyOffset(densityEnergyOffset), densityProgressVariableOffset(densityProgressVariableOffset), chemTabModel(std::move(chemTabModel)) {}

void ablate::eos::ChemTab::ChemTabSourceCalculator::AddSource(const ablate::domain::Range &cellRange, Vec locX, Vec locFVec) {
    // get access to the xArray, fArray
    PetscScalar *fArray;
    VecGetArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;
    const PetscScalar *xArray;
    VecGetArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;

    // Get the solution dm
    DM dm;
    VecGetDM(locFVec, &dm) >> utilities::PetscUtilities::checkError;

    // March over each cell in the range
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;

        // Get the current state variables for this cell
        PetscScalar *sourceAtCell = nullptr;
        DMPlexPointLocalRef(dm, iCell, fArray, &sourceAtCell) >> utilities::PetscUtilities::checkError;

        // Get the current state variables for this cell
        const PetscScalar *solutionAtCell = nullptr;
        DMPlexPointLocalRead(dm, iCell, xArray, &solutionAtCell) >> utilities::PetscUtilities::checkError;

        chemTabModel->ChemistrySource(solutionAtCell[densityOffset], solutionAtCell + densityProgressVariableOffset, sourceAtCell + densityEnergyOffset, sourceAtCell + densityProgressVariableOffset);
    }
    // cleanup
    VecRestoreArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;
}

#endif

#include "registrar.hpp"
REGISTER(ablate::eos::ChemistryModel, ablate::eos::ChemTab, "Uses a tensorflow model developed by ChemTab", ARG(std::filesystem::path, "path", "the path to the model"));
