#ifndef ABLATELIBRARY_PRESSUREFCN_HPP
#define ABLATELIBRARY_PRESSUREFCN_HPP

#include <TChem_Impl_EnthalpySpecMl.hpp>
#include "TChem_Impl_MolarWeights.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace tChemLib = TChem;

namespace ablate::eos::tChem::impl {

template <typename ValueType, typename DeviceType>
struct pressureFcn {
    using value_type = ValueType;
    using device_type = DeviceType;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    /// sacado is value type
    using value_type_1d_view_type = Tines::value_type_1d_view<value_type, device_type>;
    using kinetic_model_type = KineticModelConstData<device_type>;

    template <typename MemberType, typename KineticModelConstDataType>
    KOKKOS_INLINE_FUNCTION static value_type team_invoke(const MemberType& member,
                                                         /// input
                                                         Impl::StateVector<real_type_1d_view_type> sv_at_i,
                                                         const KineticModelConstDataType& kmcd) {
        member.team_barrier();
        // compute the Pressure
        value_type Pressure;
        TCHEM_CHECK_ERROR(!sv_at_i.isValid(), "Error: input state vector is not valid");
        {
            real_type temperature = sv_at_i.Temperature();
            real_type density = sv_at_i.Density();

            // compute the mw
            const real_type mwMix = tChemLib::Impl::MolarWeights<real_type, device_type>::team_invoke(member, sv_at_i.MassFractions(), kmcd);
            Pressure = density * kmcd.Runiv * temperature / mwMix;
        }
        return Pressure;
    }
};

}  // namespace ablate::eos::tChem::impl
#endif
