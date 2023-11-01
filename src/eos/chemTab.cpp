#include "chemTab.hpp"

#include <eos/tChem.hpp>

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

void ablate::eos::ChemTab::ChemTabModelComputeFunction(PetscReal density, const PetscReal densityProgressVariables[], PetscReal *densityEnergySource,
                                                       PetscReal *densityProgressVariableSource, PetscReal *densityMassFractions) const {
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
    std::array<TF_Tensor *, numInputs> inputValues = {nullptr};
    std::array<TF_Tensor *, numOutputs> outputValues = {nullptr, nullptr};

    std::size_t ndims = 2;

    // according to Varun this should work for including Zmix
    auto ninputs = progressVariablesNames.size();
    int64_t dims[] = {1, (int)ninputs};
    float data[ninputs];

    for (std::size_t i = 0; i < ninputs; i++) {
        data[i] = (float)(densityProgressVariables[i] / density);
    }

    std::size_t ndata = ninputs * sizeof(float);
    TF_Tensor *input_tensor = TF_NewTensor(TF_FLOAT, dims, (int)ndims, data, ndata, &NoOpDeallocator, nullptr);
    if (input_tensor == nullptr) throw std::runtime_error("ERROR: Failed TF_NewTensor");

    inputValues[0] = input_tensor;

    TF_SessionRun(session, nullptr, input.data(), inputValues.data(), (int)numInputs, output.data(), outputValues.data(), (int)numOutputs, nullptr, 0, nullptr, status);
    if (TF_GetCode(status) != TF_OK) throw std::runtime_error(TF_Message(status));
    //********** Extract source predictions

    // store physical variables (e.g. souener & mass fractions)
    float *outputArray;  // Dwyer: as counterintuitive as it may be static dependents come second, it did pass its tests!
    outputArray = (float *)TF_TensorData(outputValues[1]);
    auto p = (PetscReal)outputArray[0];
    if (densityEnergySource != nullptr) *densityEnergySource += p * density;

    // store inverted mass fractions
    if (densityMassFractions) {
        for (size_t i = 0; i < speciesNames.size(); i++) {
            densityMassFractions[i] = (PetscReal)outputArray[i + 1] * density;  // i+1 b/c i==0 is souener!
        }
    }

    // store CPV sources
    outputArray = (float *)TF_TensorData(outputValues[0]);
    if (densityProgressVariableSource != nullptr) {
        //densityProgressVariableSource[0] = 0;  // Zmix source is always 0!

        // -1 b/c we don't want to go out of bounds with the +1 below, also to prevent integer overflow
        for (size_t i = 0; i < (progressVariablesNames.size() - 1); ++i) {
            densityProgressVariableSource[i + 1] += (PetscReal)outputArray[i] * density;  // +1 b/c we are manually filling in Zmix source value (to 0)
        }
    }

    // free allocated vectors
    for (auto &t : outputValues) {
        TF_DeleteTensor(t);
    }
    for (auto &t : inputValues) {
        TF_DeleteTensor(t);
    }
}

#define safe_id(array, i) (array ? array[i] : nullptr)

void ablate::eos::ChemTab::ChemTabModelComputeFunction(const PetscReal density[], const PetscReal*const*const densityProgressVariables,
                                                       PetscReal** densityEnergySource, PetscReal** densityProgressVariableSource,
                                                       PetscReal** densityMassFractions, size_t n) const {
    // for now we are implementing batch in the same way that single calls happened
    // but testing that this works prepares the api for the real thing!
    for (size_t i=0; i<n; i++) {
        ChemTabModelComputeFunction(density[i], densityProgressVariables[i],
                                    safe_id(densityEnergySource, i),
                                    safe_id(densityProgressVariableSource, i),
                                    safe_id(densityMassFractions, i));
    }
}

void ablate::eos::ChemTab::ComputeMassFractions(std::vector<PetscReal> &progressVariables, std::vector<PetscReal> &massFractions, PetscReal density) const {
    if (progressVariables.size() != progressVariablesNames.size()) {
        throw std::invalid_argument(
                "The Progress variable size is expected to be " + std::to_string(progressVariablesNames.size()));
    }
    if (massFractions.size() != speciesNames.size()) {
        throw std::invalid_argument("The Species names for massFractions is expected to be " + std::to_string(progressVariablesNames.size()));
    }
    // the naming is wrong on purpose so that it will conform to tests.
    ComputeMassFractions(progressVariables.data(), massFractions.data(), density);
    //ComputeProgressVariables(massFractions.data(), progressVariables.data());
}

void ablate::eos::ChemTab::ComputeMassFractions(const PetscReal *densityProgressVariables, PetscReal *densityMassFractions,
                                                const PetscReal density) const {
    // call model using generalized invocation method (usable for inversion & source computation)
    ChemTabModelComputeFunction(density, densityProgressVariables, nullptr,
                                nullptr, densityMassFractions);
}

void ablate::eos::ChemTab::ComputeMassFractions(const PetscReal*const* densityProgressVariables, PetscReal** densityMassFractions,
                                                const PetscReal density[], size_t n) const {
    ChemTabModelComputeFunction(density, densityProgressVariables, nullptr,
                                nullptr, densityMassFractions, n);
}


// Batched Version
void ablate::eos::ChemTab::ComputeProgressVariables(const PetscReal *const *massFractions,
                                                    PetscReal *const *progressVariables, size_t n) const {
    for (size_t i = 0; i < n; i++) {
        ComputeProgressVariables(massFractions[i], progressVariables[i]);
    }
}

// Apparently only used for tests!
void ablate::eos::ChemTab::ComputeProgressVariables(const std::vector<PetscReal> &massFractions, std::vector<PetscReal> &progressVariables) const {
    if (progressVariables.size() != progressVariablesNames.size()) {
        throw std::invalid_argument("The Progress variable size is expected to be " + std::to_string(progressVariablesNames.size()));
    }
    if (massFractions.size() != speciesNames.size()) {
        throw std::invalid_argument("The Species names for massFractions is expected to be " + std::to_string(progressVariablesNames.size()));
    }
    ComputeProgressVariables(massFractions.data(), progressVariables.data());
}

// This is real one used elsewhere probably because it is faster
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

inline double L2_norm(PetscReal* array, int n) {
    double norm=0;
    for (int i=0; i<n; i++) {
        norm+=pow(array[i], 2);
    }
    norm = pow(norm/n, 0.5);
    return norm;
}

inline void print_array(std::string prefix, PetscReal* array, const int n) {
    std::cout << prefix;
    for (int i=0; i<n; i++) {
        std::cout << array[i] << ", ";
    }
    std::cout << std::endl;
}

void ablate::eos::ChemTab::ChemistrySource(const PetscReal density, const PetscReal densityProgressVariables[],
                                           PetscReal *densityEnergySource, PetscReal *densityProgressVariableSource) const {
    // call model using generalized invocation method (usable for inversion & source computation)
    ChemTabModelComputeFunction(density, densityProgressVariables, densityEnergySource,
                                densityProgressVariableSource, nullptr);
}

// Batched Version
void ablate::eos::ChemTab::ChemistrySource(const PetscReal*const density, const PetscReal*const*const densityProgressVariables,
                                           PetscReal** densityEnergySource, PetscReal** densityProgressVariableSource, size_t n) const {
    // call model using generalized invocation method (usable for inversion & source computation)
    ChemTabModelComputeFunction(density, densityProgressVariables, densityEnergySource,
                                densityProgressVariableSource,nullptr, n);
}

void ablate::eos::ChemTab::View(std::ostream &stream) const { stream << "EOS: " << type << std::endl; }

// How does this work? should we be overriding SourceCalc class methods or this method here?
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

ablate::eos::ThermodynamicFunction ablate::eos::ChemTab::GetThermodynamicFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    // Mask the DENSITY_YI_DECODE_FIELD for the yiField
    std::vector<domain::Field> fieldsCopy;
    for (auto &field : fields) {
        if (field.name == DENSITY_YI_DECODE_FIELD) {
            fieldsCopy.push_back(field.Rename(ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD));
        } else {
            fieldsCopy.push_back(field);
        }
    }

    // Pass the call directly into the reference eos.  It is assumed that densityYi is available
    return referenceEOS->GetThermodynamicFunction(property, fieldsCopy);
}

ablate::eos::ThermodynamicTemperatureFunction ablate::eos::ChemTab::GetThermodynamicTemperatureFunction(ablate::eos::ThermodynamicProperty property, const std::vector<domain::Field> &fields) const {
    // Mask the DENSITY_YI_DECODE_FIELD for the yiField
    std::vector<domain::Field> fieldsCopy;
    for (auto &field : fields) {
        if (field.name == DENSITY_YI_DECODE_FIELD) {
            fieldsCopy.push_back(field.Rename(ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD));
        } else {
            fieldsCopy.push_back(field);
        }
    }
    // Pass the call directly into the reference eos.  It is assumed that densityYi is available
    return referenceEOS->GetThermodynamicTemperatureFunction(property, fieldsCopy);
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

            // TODO: change for batch processing!! (ask Matt about it)
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
    } else if (finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD == field && otherProperties == std::vector<std::string>{YI, PROGRESS}) {
        // get the euler field because we need density
        auto eulerFunction = referenceEOS->GetFieldFunctionFunction(finiteVolume::CompressibleFlowFields::EULER_FIELD, property1, property2, otherProperties);

        return [=](PetscReal property1, PetscReal property2, PetscInt dim, const PetscReal velocity[], const PetscReal yiAndProgress[], PetscReal conserved[]) {
            // Compute euler
            PetscReal euler[ablate::finiteVolume::CompressibleFlowFields::RHOW + 1];  // Max size for euler
            eulerFunction(property1, property2, dim, velocity, yiAndProgress, euler);

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

            // TODO: change for batch processing!! (ask Matt about it)
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

PetscErrorCode ablate::eos::ChemTab::ComputeMassFractions(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[], PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    auto chemTab = (ablate::eos::ChemTab *)ctx;

    // hard code the field offsets
    const PetscInt EULER = 0;
    const PetscInt DENSITY_PROGRESS = 1;
    const PetscInt DENSITY_YI = 2;

    // Get the density from euler
    PetscReal density = u[uOff[EULER] + finiteVolume::CompressibleFlowFields::RHO];

    // TODO: change for batch processing!! (ask Matt about it)
    // call the compute mass fractions
    chemTab->ComputeMassFractions(u + uOff[DENSITY_PROGRESS], u + uOff[DENSITY_YI], density);

    PetscFunctionReturn(0);
}

std::vector<std::tuple<ablate::solver::CellSolver::SolutionFieldUpdateFunction, void *, std::vector<std::string>>> ablate::eos::ChemTab::GetSolutionFieldUpdates() {
    return {{ComputeMassFractions, this, {ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD, ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD, DENSITY_YI_DECODE_FIELD}}};
}
std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> ablate::eos::ChemTab::GetAdditionalFields() const {
    return {
        std::make_shared<ablate::domain::FieldDescription>(DENSITY_YI_DECODE_FIELD, DENSITY_YI_DECODE_FIELD, GetSpeciesNames(), ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM)};
}

ablate::eos::ChemTab::ChemTabSourceCalculator::ChemTabSourceCalculator(PetscInt densityOffset, PetscInt densityEnergyOffset, PetscInt densityProgressVariableOffset,
                                                                       std::shared_ptr<ChemTab> chemTabModel)
    : densityOffset(densityOffset), densityEnergyOffset(densityEnergyOffset), densityProgressVariableOffset(densityProgressVariableOffset), chemTabModel(std::move(chemTabModel)) {}

// NOTE: I'm not sure however I believe that this could be the ONLY place that needs updating for Batch processing??
// Comments seem to indicate it is the case...
void ablate::eos::ChemTab::ChemTabSourceCalculator::AddSource(const ablate::domain::Range &cellRange, Vec locX, Vec locFVec) {
    // get access to the xArray, fArray
    PetscScalar *fArray;
    VecGetArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;
    const PetscScalar *xArray;
    VecGetArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;

    // Get the solution dm
    DM dm; // NOTE: DM is topological space (i.e. grid)
    VecGetDM(locFVec, &dm) >> utilities::PetscUtilities::checkError;

    // Here we store the batched pointers needed for multiple chemTabModel->ChemistrySource() calls
    PetscInt buffer_len = cellRange.end - cellRange.start;
    //PetscScalar* allSourceAtCell[buffer_len]; // apparently these can't be used b/c of secret offsets
    //const PetscScalar* allSolutionAtCell[buffer_len]; // apparently these can't be used b/c of secret offsets
    PetscScalar allDensity[buffer_len];
    const PetscScalar* allDensityCPV[buffer_len];
    PetscScalar* allDensityEnergySource[buffer_len];
    PetscScalar* allDensityCPVSource[buffer_len];

    // TODO: change this for batch processing!!
    // March over each cell in the range
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;
        // Dwyer: iCell is the "point"

        // Def: PetscErrorCode DMPlexPointLocalRef(DM dm, PetscInt point, PetscScalar *array, void *ptr)
        // Help: return read/write access to a point in local array
        // :param array: - array to index into
        // :param ptr: output reference/return value

//        // Get the current source variables for this cell
//        DMPlexPointLocalRef(dm, iCell, fArray, &allSourceAtCell[c-cellRange.start]) >> utilities::PetscUtilities::checkError;

        PetscScalar* sourceAtCell = nullptr;
        DMPlexPointLocalRef(dm, iCell, fArray, &sourceAtCell) >> utilities::PetscUtilities::checkError;
        //assert(sourceAtCell==allSourceAtCell[c-cellRange.start]); // silly sanity check

         // Get the current state variables for this cell (CPVs)
        const PetscScalar* solutionAtCell = nullptr;
        DMPlexPointLocalRead(dm, iCell, xArray, &solutionAtCell) >> utilities::PetscUtilities::checkError;
        //allSolutionAtCell[c-cellRange.start]=solutionAtCell;
        //DMPlexPointLocalRead(dm, iCell, xArray, &allSolutionAtCell[c-cellRange.start]) >> utilities::PetscUtilities::checkError;
        //assert(solutionAtCell==allSolutionAtCell[c-cellRange.start]); // silly sanity check

        // Def: PetscErrorCode DMPlexPointLocalRead(DM dm, PetscInt point, const PetscScalar *array, void *ptr)
        // Help: return read access to a point in local array
        // NOTE: The only difference is that DMPlexPointLocalRef gives read/write access
        // & DMPlexPointLocalRead gives only ready access

        // store this cell's attributes into arg vectors
        size_t index = c-cellRange.start;
        allDensity[index]=solutionAtCell[densityOffset];
        allDensityCPV[index]=solutionAtCell + densityProgressVariableOffset;
        allDensityEnergySource[index]=sourceAtCell + densityEnergyOffset;
        allDensityCPVSource[index]=sourceAtCell + densityProgressVariableOffset;

//        // Def: ChemTab::ChemistrySource(PetscReal density, const PetscReal densityProgressVariable[], PetscReal *densityEnergySource, PetscReal *progressVariableSource)
//        // Help: last 2 (Souener & CPV_source) are the "return values"
//        chemTabModel->ChemistrySource(solutionAtCell[densityOffset], solutionAtCell + densityProgressVariableOffset,
//                                      sourceAtCell + densityEnergyOffset, sourceAtCell + densityProgressVariableOffset);
        // NOTE: These "offsets" are pointers since they are CONSTANT class attributes!
    }

    // using batch overloaded version
    chemTabModel->ChemistrySource(allDensity, allDensityCPV,allDensityEnergySource,
                                  allDensityCPVSource, buffer_len);

    // cleanup
    VecRestoreArray(locFVec, &fArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(locX, &xArray) >> utilities::PetscUtilities::checkError;
}

#endif

#include "registrar.hpp"
REGISTER(ablate::eos::ChemistryModel, ablate::eos::ChemTab, "Uses a tensorflow model developed by ChemTab", ARG(std::filesystem::path, "path", "the path to the model"));
