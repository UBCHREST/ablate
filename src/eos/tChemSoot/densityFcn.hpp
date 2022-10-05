//
// Created by klbud on 9/30/22.
//
#include <TChem_Impl_RhoMixMs.hpp>
#include "TChem_KineticModelData.hpp"
#include "eos/tChemSoot.hpp"

#ifndef ABLATE_SOOTDENSITYFCN_HPP
#define ABLATE_SOOTDENSITYFCN_HPP

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
                                                         const Impl::StateVector<real_type_1d_view_type> svGas, // Gaseous State Vector
                                                         value_type YCarbon,
                                                         const KineticModelConstDataType& kmcd) {
        member.team_barrier();
        // compute the Pressure
        value_type TotalDensity;
        TCHEM_CHECK_ERROR(!svGas.isValid(), "Error: input state vector is not valid");
        {
            value_type temperature = svGas.Temperature();
            value_type GaseousDensity = Impl::RhoMixMs<value_type,DeviceType>
                ::team_invoke(member, temperature,
                             svGas.Pressure(), svGas.MassFractions(), kmcd);
            TotalDensity = 1/((1-YCarbon)/GaseousDensity+(YCarbon/TChemSoot::solidCarbonDensity) );
        }
        return TotalDensity;
    }

    template <typename MemberType, typename KineticModelConstDataType>
    KOKKOS_INLINE_FUNCTION static value_type team_invoke(const MemberType& member,
                                                         /// input
                                                         real_type_1d_view_type TotalState,
                                                         const KineticModelConstDataType& kmcd) {
        member.team_barrier();
        real_type_1d_view GaseousState = real_type_1d_view_type("Gaseous",TChem::Impl::getStateVectorSize(kmcd.nSpec));
        TChemSoot::SplitYiState(TotalState,GaseousState,kmcd);
        const TChem::Impl::StateVector SV = TChem::Impl::StateVector(kmcd.nSpec,GaseousState);
        return team_invoke(member,SV,TotalState(kmcd.nSpec+3),kmcd);
    }
};

}  // namespace ablate::eos::tChem::impl
#endif

