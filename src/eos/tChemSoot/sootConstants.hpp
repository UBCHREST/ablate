#ifndef ABLATE_SOOTCONSTANTS_HPP
#define ABLATE_SOOTCONSTANTS_HPP

#include <TChem_Impl_RhoMixMs.hpp>
#include "TChem_KineticModelData.hpp"
#include "eos/tChemSoot.hpp"

namespace ablate::eos::tChemSoot {
//! SolidCarbonDensity
[[maybe_unused]] inline const static double solidCarbonDensity = 2000;
//! Molecular Weight of Carbon
[[maybe_unused]] inline static const double MWCarbon = 12.0107;
//! Scaling term for Ndd going into the Tines ODE Solver
[[maybe_unused]] inline static double NddScaling = 1e22;

}  // namespace ablate::eos::tChemSoot
#endif
