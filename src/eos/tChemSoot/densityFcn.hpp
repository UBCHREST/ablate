#ifndef ABLATE_SOOTDENSITYFCN_HPP
#define ABLATE_SOOTDENSITYFCN_HPP

#include <TChem_Impl_RhoMixMs.hpp>
#include "TChem_KineticModelData.hpp"
#include "eos/tChemSoot/sootConstants.hpp"
#include "stateVectorSoot.hpp"

namespace ablate::eos::tChemSoot::impl {

template <typename ValueType, typename DeviceType>
struct densityFcn {
    using value_type = ValueType;
    using device_type = DeviceType;

    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using kinetic_model_type = KineticModelConstData<device_type>;

    template <typename MemberType, typename KineticModelConstDataType>
    KOKKOS_INLINE_FUNCTION static value_type team_invoke(const MemberType& member,
                                                         /// input
                                                         StateVectorSoot<real_type_1d_view_type> totalState, const KineticModelConstDataType& kmcd) {
        member.team_barrier();
        real_type_1d_view gaseousState = real_type_1d_view_type("Gaseous", ::TChem::Impl::getStateVectorSize(kmcd.nSpec));
        ::TChem::Impl::StateVector svGas = ::TChem::Impl::StateVector(kmcd.nSpec, gaseousState);

        totalState.SplitYiState(svGas);

        value_type totalDensity;
        auto YCarbon = totalState.MassFractionCarbon();
        TCHEM_CHECK_ERROR(!svGas.isValid(), "Error: input state vector is not valid");
        {
            value_type temperature = svGas.Temperature();
            value_type gaseousDensity = Impl::RhoMixMs<value_type, DeviceType>::team_invoke(member, temperature, svGas.Pressure(), svGas.MassFractions(), kmcd);
            totalDensity = 1.0 / ((1.0 - YCarbon) / gaseousDensity + (YCarbon / ablate::eos::tChemSoot::solidCarbonDensity));
        }
        return totalDensity;
    }
};

}  // namespace ablate::eos::tChemSoot::impl
#endif
