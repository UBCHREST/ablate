#include "builder.hpp"
#include <petscviewerhdf5.h>
#include <yaml-cpp/yaml.h>
#include <utilities/petscError.hpp>
#include "flow/flow.hpp"
#include "monitors/monitor.hpp"
#include "particles/particles.hpp"
#include "solve/timeStepper.hpp"
#include "utilities/petscOptions.hpp"
#include "version.h"

void ablate::Builder::Run(std::shared_ptr<ablate::parser::Factory> parser, YAML::Node& restartNode) {
    // get the global arguments
    auto globalArguments = parser->Get(parser::ArgumentIdentifier<std::map<std::string, std::string>>{.inputName = "arguments"});
    utilities::PetscOptionsUtils::Set(globalArguments);

    // create a time stepper
    auto timeStepper = parser->Get(parser::ArgumentIdentifier<solve::TimeStepper>{.inputName = "timestepper"});

    // assume one flow field right now
    auto flow = parser->GetByName<flow::Flow>("flow");
    flow->SetupSolve(timeStepper->GetTS());

    // get the monitors from the flow factory
    auto flowMonitors = parser->GetFactory("flow")->GetByName<std::vector<monitors::Monitor>>("monitors", std::vector<std::shared_ptr<monitors::Monitor>>());
    for (auto flowMonitor : flowMonitors) {
        flowMonitor->Register(flow);
        timeStepper->AddMonitor(flowMonitor);
    }

    // get any particles that may be in the flow
    auto particleList = parser->GetByName<std::vector<particles::Particles>>("particles", std::vector<std::shared_ptr<particles::Particles>>());
    if (!particleList.empty()) {
        auto particleFactorySequence = parser->GetFactorySequence("particles");

        // initialize the flow for each
        for (std::size_t particleIndex = 0; particleIndex < particleList.size(); particleIndex++) {
            auto particle = particleList[particleIndex];
            particle->InitializeFlow(flow);

            // Get any particle monitors
            auto particleMonitors = particleFactorySequence[particleIndex]->GetByName<std::vector<monitors::Monitor>>("monitors", std::vector<std::shared_ptr<monitors::Monitor>>());
            for (auto particleMonitor : particleMonitors) {
                particleMonitor->Register(particle);
                timeStepper->AddMonitor(particleMonitor);
            }
        }
    }

    // run
    timeStepper->SetupSolve(flow);

    if (!restartNode.IsNull()) {
        TSSetStepNumber(timeStepper->GetTS(), restartNode["steps"].as<int>());
        TSSetTime(timeStepper->GetTS(), restartNode["time"].as<double>());
        TSSetTimeStep(timeStepper->GetTS(), restartNode["dt"].as<double>());

        PetscInt ranks;
        MPI_Comm_size(PETSC_COMM_WORLD, &ranks);
        std::cout << "size" << ranks << std::endl;

        // Load the vec
        auto vecPath = restartNode["solutionVec"].as<std::string>();
        PetscViewer petscViewer = nullptr;
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, vecPath.c_str(), FILE_MODE_READ, &petscViewer) >> checkError;
        //        PetscViewerHDF5Open(PETSC_COMM_WORLD, vecPath.c_str(), FILE_MODE_READ, &petscViewer) >> checkError;
        VecLoad(flow->GetSolutionVector(), petscViewer) >> checkError;
        PetscViewerDestroy(&petscViewer) >> checkError;
    }
    PetscInt vecSize;
    VecGetSize(flow->GetSolutionVector(), &vecSize);
    std::cout << "VecSize: " << vecSize << std::endl;

    timeStepper->Solve(flow);
}

void ablate::Builder::PrintVersion(std::ostream& stream) { stream << ABLATECORE_VERSION; }

void ablate::Builder::PrintInfo(std::ostream& stream) {
    stream << "ABLATE: " << std::endl;
    stream << '\t' << "Documentation: https://ablate.dev" << std::endl;
    stream << '\t' << "Source: https://github.com/UBCHREST/ablate" << std::endl;
    stream << '\t' << "Version: " << ABLATECORE_VERSION << std::endl;
}
