#include "builder.hpp"
#include "flow/flow.hpp"
#include "monitors/monitor.hpp"
#include "particles/particles.hpp"
#include "solve/timeStepper.hpp"
#include "utilities/petscOptions.hpp"
#include "version.h"

void ablate::Builder::Run(std::shared_ptr<ablate::parser::Factory> parser) {
    // get the global arguments
    auto globalArguments = parser->Get(parser::ArgumentIdentifier<std::map<std::string, std::string>>{"arguments"});
    utilities::PetscOptionsUtils::Set(globalArguments);

    // create a time stepper
    auto timeStepper = parser->Get(parser::ArgumentIdentifier<solve::TimeStepper>{"timestepper"});

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
        for (auto particleIndex = 0; particleIndex < particleList.size(); particleIndex++) {
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
    timeStepper->Solve(flow);
}

void ablate::Builder::PrintVersion(std::ostream& stream) {
    stream << "ABLATE: " << std::endl;
    stream << '\t' << "Documentation: https://ubchrest.github.io/ablate/" << std::endl;
    stream << '\t' << "Source: https://github.com/UBCHREST/ablate" << std::endl;
    stream << '\t' << "Version: " << ABLATECORE_VERSION << std::endl;
}
