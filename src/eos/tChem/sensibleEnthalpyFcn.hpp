#ifndef ABLATELIBRARY_SENSIBLEENTHALPYFCN_HPP
#define ABLATELIBRARY_SENSIBLEENTHALPYFCN_HPP

#include <TChem_Impl_EnthalpySpecMl.hpp>
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"
#include "eos/tChem.hpp"

namespace ablate::eos::tChem::impl {

template <typename ValueType, typename DeviceType>
struct SensibleEnthalpyFcn {
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
                                                         const value_type& temperature,  /// temperature
                                                         const value_type_1d_view_type& ys,
                                                         /// work space
                                                         const value_type_1d_view_type& hi, const value_type_1d_view_type& cpks,
                                                         /// const input from kinetic model
                                                         const value_type_1d_view_type& hi_ref, const KineticModelConstDataType& kmcd) {
        // compute the enthalpy of each species at temperature
        tChemLib::Impl::EnthalpySpecMlFcn<value_type, device_type>::team_invoke(member, temperature, hi, cpks, kmcd);

        Kokkos::parallel_for(Tines::RangeFactory<value_type>::TeamVectorRange(member, kmcd.nSpec), [&](const ordinal_type& i) {
            hi(i) /= kmcd.sMass(i);
            hi(i) -= hi_ref(i);
        });
        member.team_barrier();

        // compute the sensibleInternalEnergy
        value_type sensibleInternalEnergy;
        Kokkos::parallel_reduce(
            Kokkos::TeamVectorRange(member, kmcd.nSpec), [&](const ordinal_type& k, real_type& update) { update += hi(k) * ys(k); }, sensibleInternalEnergy);

        return sensibleInternalEnergy;
    }
};

}  // namespace ablate::eos::tChem::impl
#endif
