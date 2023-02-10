#ifndef ABLATE_SOOTCONSTANTS_HPP
#define ABLATE_SOOTCONSTANTS_HPP

#include <TChem_Impl_RhoMixMs.hpp>
#include "TChem_KineticModelData.hpp"
#include "eos/tChemSoot.hpp"

namespace ablate::eos::tChemSoot {
//! SolidCarbonDensity
inline const static double solidCarbonDensity = 2000;
//! Molecular Weight of Carbon
inline static const double MWCarbon = 12.0107;
//! Scaling term for Ndd going into the Tines ODE Solver
inline static double NddScaling = 1e20;

}  // namespace ablate::eos::tChem::impl
#endif

