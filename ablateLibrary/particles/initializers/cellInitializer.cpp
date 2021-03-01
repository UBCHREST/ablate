#include "cellInitializer.hpp"
#include "parser/registrar.hpp"
#include "particleCellInitializer.h"
#include "utilities/petscError.hpp"

ablate::particles::initializers::CellInitializer::CellInitializer(std::map<std::string, std::string> arguments) : Initializer(arguments) {}

void ablate::particles::initializers::CellInitializer::Initialize(ablate::flow::Flow& flow, DM particleDM) { ParticleCellInitialize(flow.GetMesh().GetDomain(), particleDM) >> ablate::checkError; }

REGISTER(ablate::particles::initializers::Initializer, ablate::particles::initializers::CellInitializer, "simple cell initializer that puts particles in every element",
         ARG(std::map<std::string TMP_COMMA std::string>, "arguments", "arguments to be passed to petsc"));