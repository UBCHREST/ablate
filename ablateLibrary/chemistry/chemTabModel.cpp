#include "chemTabModel.hpp"
#ifdef WITH_TENSORFLOW

ablate::chemistry::ChemTabModel::ChemTabModel(std::filesystem::path path) {
    const char *tags = "serve";  // default model serving tag; can change in future
    int ntags = 1;

    // check to make sure the file
    if (!exists(path)) {
        throw std::runtime_error("Cannot locate ChemTabModel Folder " + path.string());
    }
    path /= "regressor";

    // Create a new graph and status
    graph = TF_NewGraph();
    status = TF_NewStatus();

    // Prepare session
    sessionOpts = TF_NewSessionOptions();
    runOpts = NULL;

    // load and instantiate the model
    session = TF_LoadSessionFromSavedModel(sessionOpts, runOpts, path.c_str(), &tags, ntags, graph, NULL, status);
}
ablate::chemistry::ChemTabModel::~ChemTabModel() {
    TF_DeleteGraph(graph);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(sessionOpts);
    TF_DeleteStatus(status);
}
void ablate::chemistry::ChemTabModel::ChemTabModelComputeMassFractionsFunction(const PetscReal *progressVariables, PetscReal *massFractions, void *ctx) {}
void ablate::chemistry::ChemTabModel::ChemTabModelComputeSourceFunction(const PetscReal *progressVariables, PetscReal &densityEnergySource, PetscReal *progressVariableSource, void *ctx) {}
const std::vector<std::string> &ablate::chemistry::ChemTabModel::GetSpecies() const { throw std::runtime_error("not supported yet"); }
const std::vector<std::string> &ablate::chemistry::ChemTabModel::GetProgressVariables() const { throw std::runtime_error("not supported yet"); }
void ablate::chemistry::ChemTabModel::ComputeProgressVariables(const PetscReal *massFractions, PetscReal *progressVariables) const {}

#endif

#include "registrar.hpp"
REGISTER(ablate::chemistry::ChemistryModel, ablate::chemistry::ChemTabModel, "Uses a tensorflow model developed by ChemTab", ARG(std::filesystem::path, "path", "the path to the model"));
