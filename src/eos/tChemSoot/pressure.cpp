#include <Kokkos_Core.hpp>
#ifndef KOKKOS_ENABLE_CUDA

#include "eos/tChem/pressureFcn.hpp"
#include "eos/tChemSoot.hpp"
#include "pressure.hpp"

namespace tChemLib = TChem;

namespace ablate::eos::tChemSoot::impl {
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

            const StateVectorSoot<real_type_1d_view_type> sv_at_i_total(kmcd.nSpec, state_at_i);

            // Get the Gaseous State Vector
            real_type_1d_view_type state_at_i_gas = real_type_1d_view_type("Gaseous", ::TChem::Impl::getStateVectorSize(kmcd.nSpec));
            Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec, state_at_i_gas);

            // Need to scale Yi appropriately
            sv_at_i_total.SplitYiState(sv_at_i);

            sv_at_i_total.Pressure() = ablate::eos::tChem::impl::pressureFcn<real_type, device_type>::team_invoke(member, sv_at_i, kmcd);
        });
    Kokkos::Profiling::popRegion();
}

}  // namespace ablate::eos::tChemSoot::impl

[[maybe_unused]] void ablate::eos::tChemSoot::Pressure::runDeviceBatch(typename UseThisTeamPolicy<exec_space>::type& policy, const Pressure::real_type_2d_view_type& state,
                                                                       const Pressure::kinetic_model_type& kmcd) {
    ablate::eos::tChemSoot::impl::Pressure_TemplateRun("ablate::eos::tChem::Pressure::runDeviceBatch", policy, state, kmcd);
}

[[maybe_unused]] void ablate::eos::tChemSoot::Pressure::runHostBatch(typename UseThisTeamPolicy<host_exec_space>::type& policy,
                                                                     const ablate::eos::tChemSoot::Pressure::real_type_2d_view_host_type& state,
                                                                     const ablate::eos::tChemSoot::Pressure::kinetic_model_host_type& kmcd) {
    ablate::eos::tChemSoot::impl::Pressure_TemplateRun("ablate::eos::tChem::Pressure::runHostBatch", policy, state, kmcd);
}
#endif