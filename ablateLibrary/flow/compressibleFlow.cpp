#include "compressibleFlow.hpp"
#include <utilities/mpiError.hpp>
#include "compressibleFlow.h"
#include "flow/fluxDifferencer/ausmFluxDifferencer.hpp"
#include "utilities/petscError.hpp"
#include "fvSupport.h"

static const char* compressibleFlowComponentNames[TOTAL_COMPRESSIBLE_FLOW_COMPONENTS + 1] = {"rho", "rhoE", "rhoU", "rhoV", "rhoW", "unknown"};
static const char* compressibleAuxComponentNames[TOTAL_COMPRESSIBLE_AUX_COMPONENTS + 1] = {"T", "vel", "unknown"};

static PetscErrorCode UpdateAuxTemperatureField(FlowData_CompressibleFlow flowParameters, PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscScalar* conservedValues,
                                                PetscScalar* auxField) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[RHO];
    PetscReal totalEnergy = conservedValues[RHOE] / density;

    PetscErrorCode ierr = flowParameters->computeTemperatureFunction(NULL, dim, density, totalEnergy, conservedValues + RHOU, &auxField[T], flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode UpdateAuxVelocityField(FlowData_CompressibleFlow flowData, PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscScalar* conservedValues,
                                             PetscScalar* auxField) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[RHO];

    for (PetscInt d = 0; d < dim; d++) {
        auxField[d] = conservedValues[RHOU + d] / density;
    }

    PetscFunctionReturn(0);
}

ablate::flow::CompressibleFlow::CompressibleFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<eos::EOS> eosIn, std::shared_ptr<parameters::Parameters> parameters,
                                                 std::shared_ptr<fluxDifferencer::FluxDifferencer> fluxDifferencerIn, std::shared_ptr<parameters::Parameters> options,
                                                 std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization,
                                                 std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                                                 std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolutions)
    : Flow(name, mesh, parameters, options, initialization, boundaryConditions, {}, exactSolutions),
      eos(eosIn),
      fluxDifferencer(fluxDifferencerIn == nullptr ? std::make_shared<fluxDifferencer::AusmFluxDifferencer>() : fluxDifferencerIn) {
    // Create a compressibleFlowData
    PetscNew(&compressibleFlowData);

    // make sure that the dm works with fv
    const PetscInt ghostCellDepth = 1;
    DM& dm = this->dm->GetDomain();
    {  // Make sure that the flow is setup distributed
        DM dmDist;
        DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE) >> checkError;
        DMPlexDistribute(dm, ghostCellDepth, NULL, &dmDist) >> checkError;
        if (dmDist) {
            DMDestroy(&dm) >> checkError;
            dm = dmDist;
        }
    }

    // create any ghost cells that are needed
    {
        DM gdm;
        DMPlexConstructGhostCells(dm, NULL, NULL, &gdm) >> checkError;
        DMDestroy(&dm) >> checkError;
        dm = gdm;
    }

    // Copy over the application context if needed
    DMSetApplicationContext(dm, this) >> checkError;

    // Register a single field
    PetscInt numberComponents = 2 + dim;
    RegisterField({.fieldName = "euler", .fieldPrefix = "euler", .components = numberComponents, .fieldType = FieldType::FV});
//    RegisterField({.fieldName = "yi", .fieldPrefix = "yi", .components = 2, .fieldType = FieldType::FV});

    FinalizeRegisterFields();

    // register the required auxFields
    RegisterAuxField({.fieldName = compressibleAuxComponentNames[T], .fieldPrefix = compressibleAuxComponentNames[T], .components = 1, .fieldType = FieldType::FV});
    RegisterAuxField({.fieldName = compressibleAuxComponentNames[VEL], .fieldPrefix = compressibleAuxComponentNames[VEL], .components = dim, .fieldType = FieldType::FV});

    // Start problem setup
    PetscDS prob;
    DMGetDS(dm, &prob) >> checkError;

    // Set up each field
    PetscInt eulerField = 0;
    PetscDSSetContext(prob, eulerField, compressibleFlowData) >> checkError;
//    diffusionCalculationFunctions.push_back(CompressibleFlowEulerDiffusion);

    rhsFunctionDescriptions.push_back(FVMRHSFunctionDescription{
        .function = CompressibleFlowComputeEulerFlux,
        .context = compressibleFlowData,
        .field = 0,
        .inputFields = {0, 0, 0, 0},
        .numberInputFields = 1,
        .auxFields = {0, 0, 0, 0},
        .numberAuxFields = 0
    });


    rhsFunctionDescriptions.push_back(FVMRHSFunctionDescription{
        .function = CompressibleFlowEulerDiffusion,
        .context = compressibleFlowData,
        .field = 0,
        .inputFields = {0, 0, 0, 0},
        .numberInputFields = 1,
        .auxFields = {0, 1, 0, 0},
        .numberAuxFields = 2
    });

//    PetscDSSetRiemannSolver(prob, eulerField, CompressibleFlowComputeEulerFlux) >> checkError;

    // if there are species
//    if(!eos->GetSpecies().empty()){
//        PetscInt massFracField = 1;
//        PetscDSSetContext(prob, massFracField, compressibleFlowData) >> checkError;
//    }

    // Set the flux calculator solver for each component
    PetscDSSetFromOptions(prob) >> checkError;

    // Store the required data for the low level c functions
    compressibleFlowData->cfl = parameters->Get<PetscReal>("cfl", 0.5);
    compressibleFlowData->mu = parameters->Get<PetscReal>("mu", 0.0);
    compressibleFlowData->k = parameters->Get<PetscReal>("k", 0.0);

    // extract the difference function from fluxDifferencer object
    compressibleFlowData->fluxDifferencer = fluxDifferencer->GetFluxDifferencerFunction();

    // set the decode state function
    compressibleFlowData->decodeStateFunction = eos->GetDecodeStateFunction();
    compressibleFlowData->decodeStateFunctionContext = eos->GetDecodeStateContext();
    compressibleFlowData->computeTemperatureFunction = eos->GetComputeTemperatureFunction();
    compressibleFlowData->computeTemperatureContext = eos->GetComputeTemperatureContext();

    // Set the update fields
    auxFieldUpdateFunctions[T] = UpdateAuxTemperatureField;
    auxFieldUpdateFunctions[VEL] = UpdateAuxVelocityField;

    // PetscErrorCode PetscOptionsGetBool(PetscOptions options,const char pre[],const char name[],PetscBool *ivalue,PetscBool *set)
    compressibleFlowData->automaticTimeStepCalculator = PETSC_TRUE;
    PetscOptionsGetBool(NULL, NULL, "-automaticTimeStepCalculator", &(compressibleFlowData->automaticTimeStepCalculator), NULL);
}

ablate::flow::CompressibleFlow::~CompressibleFlow() { PetscFree(compressibleFlowData); }

void ablate::flow::CompressibleFlow::ComputeTimeStep(TS ts, ablate::flow::Flow& flow) {
    PetscFunctionBeginUser;
    // Get the dm and current solution vector
    DM dm;
    TSGetDM(ts, &dm) >> checkError;
    Vec v;
    TSGetSolution(ts, &v) >> checkError;

    // Get the flow param
    ablate::flow::CompressibleFlow& compressibleFlow = dynamic_cast<ablate::flow::CompressibleFlow&>(flow);
    FlowData_CompressibleFlow flowParameters = compressibleFlow.compressibleFlowData;

    // Get the fv geom
    PetscReal minCellRadius;
    DMPlexGetGeometryFVM(dm, NULL, NULL, &minCellRadius) >> checkError;
    PetscInt cStart, cEnd;
    DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd) >> checkError;
    const PetscScalar* x;
    VecGetArrayRead(v, &x) >> checkError;

    // Get the dim from the dm
    PetscInt dim;
    DMGetDimension(dm, &dim) >> checkError;

    // assume the smallest cell is the limiting factor for now
    const PetscReal dx = 2.0 * minCellRadius;

    // March over each cell
    PetscReal dtMin = 1000.0;
    for (PetscInt c = cStart; c < cEnd; ++c) {
        const PetscReal* xc;
        DMPlexPointGlobalFieldRead(dm, c, 0, x, &xc) >> checkError;

        if (xc) {  // must be real cell and not ghost
            PetscReal rho = xc[RHO];
            PetscReal vel[3];
            for (PetscInt i = 0; i < dim; i++) {
                vel[i] = xc[RHOU + i] / rho;
            }

            // Get the speed of sound from the eos
            PetscReal ie;
            PetscReal a;
            PetscReal p;
            flowParameters->decodeStateFunction(NULL, dim, rho, xc[RHOE] / rho, vel, &ie, &a, &p, flowParameters->decodeStateFunctionContext) >> checkError;

            PetscReal u = xc[RHOU] / rho;
            PetscReal dt = flowParameters->cfl * dx / (a + PetscAbsReal(u));
            dtMin = PetscMin(dtMin, dt);
        }
    }
    PetscInt rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank);

    PetscReal dtMinGlobal;
    MPI_Allreduce(&dtMin, &dtMinGlobal, 1, MPIU_REAL, MPI_MIN, PetscObjectComm((PetscObject)ts)) >> checkMpiError;

    TSSetTimeStep(ts, dtMinGlobal) >> checkError;

    if (PetscIsNanReal(dtMinGlobal)) {
        throw std::runtime_error("Invalid timestep selected for flow");
    }

    VecRestoreArrayRead(v, &x) >> checkError;
}

PetscErrorCode ablate::flow::CompressibleFlow::CompressibleFlowRHSFunctionLocal(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void* ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    ablate::flow::CompressibleFlow* flow = (ablate::flow::CompressibleFlow*)ctx;

    if (flow->compressibleFlowData->k || flow->compressibleFlowData->mu) {
        // update any aux fields
        ierr = FVFlowUpdateAuxFieldsFV(flow->dm->GetDomain(), flow->auxDM, time, locXVec, flow->auxField, TOTAL_COMPRESSIBLE_AUX_COMPONENTS, flow->auxFieldUpdateFunctions, flow->compressibleFlowData);
        CHKERRQ(ierr);
    }

    // compute the euler flux across each face (note CompressibleFlowComputeEulerFlux has already been registered)
    ierr = ABLATE_DMPlexTSComputeRHSFunctionFVM(&flow->rhsFunctionDescriptions[0], flow->rhsFunctionDescriptions.size(), dm, time, locXVec, globFVec, &flow->compressibleFlowData);
    CHKERRQ(ierr);
//
//    // if there are any coefficients for diffusion, compute diffusion
//    if (flow->compressibleFlowData->k || flow->compressibleFlowData->mu) {
//        // update any aux fields
//        ierr = FVFlowUpdateAuxFieldsFV(flow->dm->GetDomain(), flow->auxDM, time, locXVec, flow->auxField, TOTAL_COMPRESSIBLE_AUX_COMPONENTS, flow->auxFieldUpdateFunctions, flow->compressibleFlowData);
//        CHKERRQ(ierr);
//
//        // compute the RHS sources
//        ierr = CompressibleFlowDiffusionSourceRHSFunctionLocal(dm, flow->auxDM, time, locXVec, flow->auxField, globFVec, flow->compressibleFlowData, &flow->diffusionCalculationFunctions[0]);
//        CHKERRQ(ierr);
//    }

    PetscFunctionReturn(0);
}

void ablate::flow::CompressibleFlow::CompleteProblemSetup(TS ts) {
    Flow::CompleteProblemSetup(ts);

    if (compressibleFlowData->automaticTimeStepCalculator) {
        preStepFunctions.push_back(ComputeTimeStep);
    }

    // Override the DMTSSetRHSFunctionLocal in DMPlexTSComputeRHSFunctionFVM with a function that includes euler and diffusion source terms
    DMTSSetRHSFunctionLocal(dm->GetDomain(), CompressibleFlowRHSFunctionLocal, this) >> checkError;

    // copy over any boundary information from the dm, to the aux dm and set the sideset
    if (auxDM) {
        PetscDS flowProblem;
        DMGetDS(dm->GetDomain(), &flowProblem) >> checkError;
        PetscDS auxProblem;
        DMGetDS(auxDM, &auxProblem) >> checkError;

        // Get the number of boundary conditions and other info
        PetscInt numberBC;
        PetscDSGetNumBoundary(flowProblem, &numberBC) >> checkError;
        PetscInt numberAuxFields;
        PetscDSGetNumFields(auxProblem, &numberAuxFields) >> checkError;

        for (PetscInt bc = 0; bc < numberBC; bc++) {
            DMBoundaryConditionType type;
            const char* name;
            const char* labelName;
            PetscInt field;
            PetscInt numberIds;
            const PetscInt* ids;

            // Get the boundary
            PetscDSGetBoundary(flowProblem, bc, &type, &name, &labelName, &field, NULL, NULL, NULL, NULL, &numberIds, &ids, NULL) >> checkError;

            // If this is for euler and DM_BC_NATURAL_RIEMANN add it to the aux
            if (type == DM_BC_NATURAL_RIEMANN && field == 0) {
                for (PetscInt af = 0; af < numberAuxFields; af++) {
                    PetscDSAddBoundary(auxProblem, type, name, labelName, af, 0, NULL, NULL, NULL, numberIds, ids, NULL) >> checkError;
                }
            }
        }
    }
}
void ablate::flow::CompressibleFlow::CompleteFlowInitialization(DM, Vec) {}

#include "parser/registrar.hpp"
REGISTER(ablate::flow::Flow, ablate::flow::CompressibleFlow, "compressible finite volume flow", ARG(std::string, "name", "the name of the flow field"),
         ARG(ablate::mesh::Mesh, "mesh", "the  mesh and discretization"), ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"),
         ARG(ablate::parameters::Parameters, "parameters", "the compressible flow parameters cfl, gamma, etc."),
         OPT(ablate::flow::fluxDifferencer::FluxDifferencer, "fluxDifferencer", "the flux differencer (defaults to AUSM)"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSc"), OPT(std::vector<mathFunctions::FieldSolution>, "initialization", "the flow field initialization"),
         OPT(std::vector<flow::boundaryConditions::BoundaryCondition>, "boundaryConditions", "the boundary conditions for the flow field"),
         OPT(std::vector<mathFunctions::FieldSolution>, "exactSolution", "optional exact solutions that can be used for error calculations"));