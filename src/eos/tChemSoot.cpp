#include "tChemSoot.hpp"

ablate::eos::TChemSoot::TChemSoot(std::filesystem::path mechanismFileIn, std::filesystem::path thermoFileIn) :TChem( mechanismFileIn , thermoFileIn)  {}

#include "registrar.hpp"
REGISTER(ablate::eos::EOS, ablate::eos::TChemSoot, "[TChemV2](https://github.com/sandialabs/TChem) ideal gas eos modified with an LHF formulation for soot implementation", ARG(std::filesystem::path, "mechFile", "the mech file (CHEMKIN Format or Cantera Yaml)"),
         OPT(std::filesystem::path, "thermoFile", "the thermo file (CHEMKIN Format if mech file is CHEMKIN)"));