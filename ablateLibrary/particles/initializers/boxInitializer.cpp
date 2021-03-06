#include "boxInitializer.hpp"
#include "parser/registrar.hpp"
#include "particleBoxInitializer.h"
#include "utilities/petscError.hpp"

ablate::particles::initializers::BoxInitializer::BoxInitializer(std::map<std::string, std::string> arguments) : Initializer(arguments) {}

void ablate::particles::initializers::BoxInitializer::Initialize(ablate::flow::Flow& flow, DM particleDM) { ParticleBoxInitialize(flow.GetMesh().GetDomain(), particleDM) >> ablate::checkError; }

REGISTER(ablate::particles::initializers::Initializer, ablate::particles::initializers::BoxInitializer, "simple box initializer that puts particles in a defined box",
         ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"));