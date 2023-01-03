#include "chemTabModel.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#ifdef WITH_TENSORFLOW

void NoOpDeallocator(void *data, size_t a, void *b) {}

ablate::chemistry::ChemTabModel::ChemTabModel(std::filesystem::path path) {
    const char *tags = "serve";  // default model serving tag; can change in future
    int ntags = 1;

    // check to make sure the file
    if (!exists(path)) {
        throw std::runtime_error("Cannot locate ChemTabModel Folder " + path.string());
    }
    const std::string rpath = path / "regressor";
    const std::string wpath = path / "weights.csv";
    const std::string ipath = path / "weights_inv.csv";

    // Check for missing files
    if (!std::filesystem::exists(rpath)) {
        throw std::runtime_error(
            "The 'regressor' file cannot be located in the "
            "specified ChemTabModel Folder " +
            path.string());
    }
    if (!std::filesystem::exists(wpath)) {
        throw std::runtime_error(
            "The 'weights.csv' file cannot be located in the "
            "specified ChemTabModel Folder " +
            path.string());
    }
    if (!std::filesystem::exists(ipath)) {
        throw std::runtime_error(
            "The 'weights_inv.csv' file cannot be located in "
            "the specified ChemTabModel Folder " +
            path.string());
    }

    // Load the source energy predictor model first
    graph = TF_NewGraph();
    status = TF_NewStatus();
    sessionOpts = TF_NewSessionOptions();
    runOpts = NULL;
    session = TF_LoadSessionFromSavedModel(sessionOpts, runOpts, rpath.c_str(), &tags, ntags, graph, NULL, status);

    std::fstream inputFileStream;
    // load the meta data from the weights.csv file
    inputFileStream.open(wpath.c_str(), std::ios::in);
    ExtractMetaData(inputFileStream);
    inputFileStream.close();
    // load the basis vectors from the weights.csv and weights_inv.csv files
    // first allocate memory for both weight matrices
    Wmat = (PetscReal **)malloc(speciesNames.size() * sizeof(PetscReal *));
    iWmat = (PetscReal **)malloc(progressVariablesNames.size() * sizeof(PetscReal *));
    for (std::size_t i = 0; i < speciesNames.size(); i++) {
        Wmat[i] = (PetscReal *)malloc(progressVariablesNames.size() * sizeof(PetscReal));
    }
    for (std::size_t i = 0; i < progressVariablesNames.size(); i++) {
        iWmat[i] = (PetscReal *)malloc(speciesNames.size() * sizeof(PetscReal));
    }
    inputFileStream.open(wpath.c_str(), std::ios::in);
    LoadBasisVectors(inputFileStream, progressVariablesNames.size() - 1, Wmat);
    inputFileStream.close();
    inputFileStream.open(ipath.c_str(), std::ios::in);
    LoadBasisVectors(inputFileStream, speciesNames.size(), iWmat);
    inputFileStream.close();
}

ablate::chemistry::ChemTabModel::~ChemTabModel() {
    TF_DeleteGraph(graph);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(sessionOpts);
    TF_DeleteStatus(status);
    free(sourceEnergyScaler);
    for (std::size_t i = 0; i < speciesNames.size(); i++) free(Wmat[i]);
    for (std::size_t i = 0; i < progressVariablesNames.size(); i++) free(iWmat[i]);

    free(Wmat);
    free(iWmat);
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
}

void ablate::chemistry::ChemTabModel::ExtractMetaData(std::istream &inputStream) {
    // determine the headers from the first row
    std::string line;
    std::getline(inputStream, line);
    int i = 0;

    // push the Zmix name as the first "progress variables" name
    progressVariablesNames.push_back("zmix");
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

void ablate::chemistry::ChemTabModel::LoadBasisVectors(std::istream &inputStream, std::size_t cols, double **W) {
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

void ablate::chemistry::ChemTabModel::ChemTabModelComputeFunction(const PetscReal progressVariables[], const std::size_t progressVariablesSize, PetscReal *predictedSourceEnergy,
                                                                  PetscReal *progressVariableSource, const std::size_t progressVariableSourceSize, PetscReal *massFractions,
                                                                  std::size_t massFractionsSize, void *ctx) {
    auto ctModel = (ablate::chemistry::ChemTabModel *)ctx;
    // size of progressVariables should match the expected number of
    // progressVariables
    if (progressVariablesSize != ctModel->progressVariablesNames.size()) {
        throw std::invalid_argument(
            "The progressVariables size does not match the "
            "supported number of progressVariables");
    }
    //********* Get Input tensor
    int numInputs = 1;
    TF_Output *input = (TF_Output *)malloc(sizeof(TF_Output) * numInputs);
    TF_Output t0 = {TF_GraphOperationByName(ctModel->graph, "serving_default_input_1"), 0};

    if (t0.oper == NULL) throw std::runtime_error("ERROR: Failed TF_GraphOperationByName serving_default_input_1");
    input[0] = t0;
    //********* Get Output tensor
    int numOutputs = 2;
    TF_Output *output = (TF_Output *)malloc(sizeof(TF_Output) * numOutputs);

    TF_Output t_sourceenergy = {TF_GraphOperationByName(ctModel->graph, "StatefulPartitionedCall"), 0};
    TF_Output t_sourceterms = {TF_GraphOperationByName(ctModel->graph, "StatefulPartitionedCall"), 1};

    if (t_sourceenergy.oper == NULL) throw std::runtime_error("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall:0");
    if (t_sourceterms.oper == NULL) throw std::runtime_error("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall:1");
    output[0] = t_sourceenergy;
    output[1] = t_sourceterms;
    //********* Allocate data for inputs & outputs
    TF_Tensor **inputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * numInputs);
    TF_Tensor **outputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * numOutputs);

    int ndims = 2;

    // according to Varun this should work for including Zmix
    int ninputs = (int)ctModel->progressVariablesNames.size();
    int64_t dims[] = {1, ninputs};
    float data[ninputs];
    // Ignoring the zmix variable for predicting the source terms
    for (int i = 0; i < ninputs; i++) {
        data[i] = progressVariables[i];
    }

    int ndata = ninputs * sizeof(float);
    TF_Tensor *input_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
    if (input_tensor == NULL) throw std::runtime_error("ERROR: Failed TF_NewTensor");

    inputValues[0] = input_tensor;

    TF_SessionRun(ctModel->session, NULL, input, inputValues, numInputs, output, outputValues, numOutputs, NULL, 0, NULL, ctModel->status);
    if (TF_GetCode(ctModel->status) != TF_OK) throw std::runtime_error(TF_Message(ctModel->status));
    //********** Extract source predictions

    // store physical variables (e.g. souener & mass fractions)
    float *outputArray;  // Dwyer: as counter intuitive as it may be static dependents come second, it did pass its tests!
    outputArray = (float *)TF_TensorData(outputValues[1]);
    PetscReal p = (PetscReal)outputArray[0];
    if (predictedSourceEnergy != NULL) *predictedSourceEnergy = p;

    // store inverted mass fractions
    for (size_t i = 0; i < massFractionsSize; i++) {
        massFractions[i] = (PetscReal)outputArray[i + 1];  // i+1 b/c i==0 is souener!
    }

    // store CPV sources
    outputArray = (float *)TF_TensorData(outputValues[0]);
    progressVariableSource[0] = 0;  // Zmix source is always 0!
    for (size_t i = 0; i < progressVariableSourceSize; i++) {
        progressVariableSource[i + 1] = (PetscReal)outputArray[i];  // +1 b/c we are manually filling in Zmix source value (to 0)
    }
    // free allocated vectors
    free(inputValues);
    free(outputValues);
    free(input);
    free(output);
}

// void ablate::chemistry::ChemTabModel::ChemTabModelComputeMassFractionsFunction(const PetscReal *progressVariables, std::size_t progressVariablesSize, PetscReal *massFractions,
//                                                                               std::size_t massFractionsSize, void *ctx) {
//    // y = inv(W)'C
//    // for now the mass fractions will be obtained using the inverse of the
//    // weights. Will be replaced by a ML predictive model in the next iteration
//    auto ctModel = (ChemTabModel *)ctx;
//    // size of progressVariables should match the expected number of
//    // progressVariables
//    if (progressVariablesSize != ctModel->progressVariablesNames.size()) {
//        throw std::invalid_argument(
//            "The progressVariables size does not match the "
//            "supported number of progressVariables");
//    }
//    // size of massFractions should match the expected number of species
//    if (massFractionsSize != ctModel->speciesNames.size()) {
//        throw std::invalid_argument(
//            "The massFractions size does not match the "
//            "supported number of species");
//    }
//    for (size_t i = 0; i < ctModel->speciesNames.size(); i++) {
//        PetscReal v = 0;
//        // j starts from 1 because the first entry in progressVariables is assumed
//        // to be zMix
//        for (size_t j = 1; j < ctModel->progressVariablesNames.size(); j++) {
//            v += ctModel->iWmat[j - 1][i] * progressVariables[j];
//        }
//        massFractions[i] = v;
//    }
//}

void ablate::chemistry::ChemTabModel::ChemTabModelComputeMassFractionsFunction(const PetscReal *progressVariables, std::size_t progressVariablesSize, PetscReal *massFractions,
                                                                               std::size_t massFractionsSize, void *ctx) {
    auto ctModel = (ChemTabModel *)ctx;
    // size of massFractions should match the expected number of species
    if (massFractionsSize != ctModel->speciesNames.size()) {
        throw std::invalid_argument(
            "The massFractions size does not match the "
            "supported number of species");
    }

    // call model using generalized invokation method (usable for inversion & source computation)
    ChemTabModelComputeFunction(progressVariables, progressVariablesSize, NULL, NULL, 0, massFractions, massFractionsSize, ctx);
}

void ablate::chemistry::ChemTabModel::ChemTabModelComputeSourceFunction(const PetscReal progressVariables[], const std::size_t progressVariablesSize, PetscReal *predictedSourceEnergy,
                                                                        PetscReal *progressVariableSource, const std::size_t progressVariableSourceSize, void *ctx) {
    auto ctModel = (ChemTabModel *)ctx;
    // size of progressVariableSource should match the expected number of progressVariables (excluding zmix)
    if (progressVariableSourceSize != ctModel->progressVariablesNames.size()) {
        throw std::invalid_argument("The progressVariableSource size does not match the supported number of progressVariables");
    }

    // call model using generalized invokation method (usable for inversion & source computation)
    ChemTabModelComputeFunction(progressVariables, progressVariablesSize, predictedSourceEnergy, progressVariableSource, progressVariableSourceSize, NULL, 0, ctx);
}
// void ChemTabModelComputeFunction(const PetscReal progressVariables[], const std::size_t progressVariablesSize, PetscReal *predictedSourceEnergy,
//                                 PetscReal *progressVariableSource, const std::size_t progressVariableSourceSize,
//                                 PetscReal *massFractions, std::size_t massFractionsSize, void *ctx) {

const std::vector<std::string> &ablate::chemistry::ChemTabModel::GetSpecies() const { return speciesNames; }

const std::vector<std::string> &ablate::chemistry::ChemTabModel::GetProgressVariables() const { return progressVariablesNames; }
void ablate::chemistry::ChemTabModel::ComputeProgressVariables(const PetscReal *massFractions, std::size_t massFractionsSize, PetscReal *progressVariables, std::size_t progressVariablesSize) const {
    // c = W'y
    // size of progressVariables should match the expected number of
    // progressVariables
    if (progressVariablesSize != progressVariablesNames.size()) {
        throw std::invalid_argument(
            "The progressVariables size does not match the "
            "supported number of progressVariables");
    }
    // size of massFractions should match the expected number of species
    if (massFractionsSize != speciesNames.size()) {
        throw std::invalid_argument(
            "The massFractions size does not match the "
            "supported number of species");
    }
    // the first entry in progressVariables corresponds to zMix and is fixed to 0
    // progressVariables[0] = 0;
    for (size_t i = 0; i < progressVariablesNames.size(); i++) {
        PetscReal v = 0;
        for (size_t j = 0; j < speciesNames.size(); j++) {
            v += Wmat[j][i] * massFractions[j];
        }
        progressVariables[i] = v;
    }
}

#endif

#include "registrar.hpp"
REGISTER(ablate::chemistry::ChemistryModel, ablate::chemistry::ChemTabModel, "Uses a tensorflow model developed by ChemTab", ARG(std::filesystem::path, "path", "the path to the model"));
