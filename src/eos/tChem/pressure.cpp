#include "pressure.hpp"
#include "TChem_Impl_MolarWeights.hpp"

namespace tChemLib = TChem;

namespace ablate::eos::tChem::impl {
template <typename PolicyType, typename DeviceType>
void Pressure_TemplateRun(const std::string& profile_name,
                          /// team size setting
                          const PolicyType& policy, const Tines::value_type_2d_view<real_type, DeviceType>& state, const KineticModelConstData<DeviceType>& kmcd) {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;

    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;

    Kokkos::parallel_for(
        profile_name, policy, KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const ordinal_type i = member.league_rank();
            const real_type_1d_view_type state_at_i = Kokkos::subview(state, i, Kokkos::ALL());

            const Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec, state_at_i);
            TCHEM_CHECK_ERROR(!sv_at_i.isValid(), "Error: input state vector is not valid");
            {
                real_type temperature = sv_at_i.Temperature();
                real_type density = sv_at_i.Density();

                // compute the mw
                const real_type mwMix = tChemLib::Impl::MolarWeights<real_type, device_type>::team_invoke(member, sv_at_i.MassFractions(), kmcd);
                sv_at_i.Pressure() = density * kmcd.Runiv * temperature / mwMix;
            }
        });
    Kokkos::Profiling::popRegion();
}

}  // namespace ablate::eos::tChem::impl

[[maybe_unused]] void ablate::eos::tChem::Pressure::runDeviceBatch(typename UseThisTeamPolicy<exec_space>::type& policy, const Pressure::real_type_2d_view_type& state, const Pressure::kinetic_model_type& kmcd) {
    ablate::eos::tChem::impl::Pressure_TemplateRun("ablate::eos::tChem::Pressure::runDeviceBatch", policy, state, kmcd);
}

[[maybe_unused]] void ablate::eos::tChem::Pressure::runHostBatch(typename UseThisTeamPolicy<host_exec_space>::type& policy, const ablate::eos::tChem::Pressure::real_type_2d_view_host_type& state,
                                                const ablate::eos::tChem::Pressure::kinetic_model_host_type& kmcd) {
    ablate::eos::tChem::impl::Pressure_TemplateRun("ablate::eos::tChem::Pressure::runHostBatch", policy, state, kmcd);
}
