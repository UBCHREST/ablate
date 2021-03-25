#include "tracer.hpp"
#include "parser/registrar.hpp"
#include "particleTracer.h"
#include "solve/timeStepper.hpp"
#include "utilities/petscError.hpp"

ablate::particles::Tracer::Tracer(std::string name, int ndims, std::map<std::string, std::string> arguments, std::shared_ptr<particles::initializers::Initializer> initializer,
                                  std::shared_ptr<mathFunctions::MathFunction> exactSolution)
    : Particles(name, arguments, initializer) {
    ParticleTracerCreate(&particleData, ndims) >> checkError;

    // set the exact solution
    SetExactSolution(exactSolution);
}

ablate::particles::Tracer::~Tracer() { ParticleTracerDestroy(&particleData); }
void ablate::particles::Tracer::InitializeFlow(std::shared_ptr<flow::Flow> flow, std::shared_ptr<solve::TimeStepper> timeStepper) {
    // Call the base to initialize the flow
    Particles::InitializeFlow(flow, timeStepper);

    // call tracer specific setup
    ParticleTracerSetupIntegrator(particleData, particleTs, flow->GetFlowData()) >> checkError;
}

REGISTER(ablate::particles::Particles, ablate::particles::Tracer, "massless particles that advect with the flow", ARG(std::string, "name", "the name of the particle group"),
         ARG(int, "ndims", "the number of dimensions for the particle"), ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"),
         ARG(particles::initializers::Initializer, "initializer", "the initial particle setup methods"), ARG(mathFunctions::MathFunction, "exactSolution", "the particle location exact solution"));