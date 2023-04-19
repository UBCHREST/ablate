#include <Kokkos_Macros.hpp>
#ifndef KOKKOS_ENABLE_CUDA
#include "temperature.hpp"

#include "eos/tChemSoot/temperatureFcn.hpp"

namespace ablate::eos::tChemSoot::impl {
template <typename PolicyType, typename DeviceType>
void Temperature_TemplateRun(const std::string& profile_name,
                             /// team size setting
                             const PolicyType& policy,
                             /// Total State Vector
                             const Tines::value_type_2d_view<real_type, DeviceType>& state,
                             /// Total Internal Energy
                             const Tines::value_type_1d_view<real_type, DeviceType>& internalEnergyRef,
                             /// Species Scratch Vector
                             const Tines::value_type_2d_view<real_type, DeviceType>& enthalpyMass,
                             /// Reference enthalpyies or all species
                             const Tines::value_type_1d_view<real_type, DeviceType>& enthalpyReference,
                             /// kinetics gas model
                             const KineticModelConstData<DeviceType>& kmcd) {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;

    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;

    Kokkos::parallel_for(
        profile_name, policy, KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const ordinal_type i = member.league_rank();
            const real_type_1d_view_type state_at_i = Kokkos::subview(state, i, Kokkos::ALL());
            const StateVectorSoot<real_type_1d_view_type> sv_at_i_total(kmcd.nSpec, state_at_i);

            const real_type_1d_view_type hi_at_i = Kokkos::subview(enthalpyMass, i, Kokkos::ALL());
            const real_type_0d_view_type internalEnergyRef_at_i = Kokkos::subview(internalEnergyRef, i);

            // Pull out Carbon Mass Fractions
            const real_type Yc = sv_at_i_total.MassFractionCarbon();
            // Create the Gaseous State Vector
            real_type_1d_view_type state_at_i_gas = real_type_1d_view_type("Gaseous", ::TChem::Impl::getStateVectorSize(kmcd.nSpec));
            Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec, state_at_i_gas);
            sv_at_i_total.SplitYiState(sv_at_i);

            const ordinal_type level = 1;
            const ordinal_type per_team_extent = Temperature::getWorkSpaceSize(kmcd.nSpec);
            //        using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;
            Scratch<real_type_1d_view_type> work(member.team_scratch(level), per_team_extent);
            auto cpks = real_type_1d_view_type((real_type*)work.data(), kmcd.nSpec);
            sv_at_i_total.Temperature() =
                ablate::eos::tChemSoot::impl::TemperatureFcn<real_type, device_type>::team_invoke(member, sv_at_i, Yc, internalEnergyRef_at_i, hi_at_i, cpks, enthalpyReference, kmcd);
        });

    Kokkos::Profiling::popRegion();
}

}  // namespace ablate::eos::tChemSoot::impl

[[maybe_unused]] void ablate::eos::tChemSoot::Temperature::runDeviceBatch(typename UseThisTeamPolicy<exec_space>::type& policy, const Temperature::real_type_2d_view_type& state,
                                                                          const Temperature::real_type_1d_view_type& internalEnergyRef, const Temperature::real_type_2d_view_type& enthalpyMass,
                                                                          const Temperature::real_type_1d_view_type& enthalpyReference, const Temperature::kinetic_model_type& kmcd) {
    ablate::eos::tChemSoot::impl::Temperature_TemplateRun("ablate::eos::tChemSoot::Temperature::runDeviceBatch", policy, state, internalEnergyRef, enthalpyMass, enthalpyReference, kmcd);
}

[[maybe_unused]] void ablate::eos::tChemSoot::Temperature::runHostBatch(const typename UseThisTeamPolicy<host_exec_space>::type& policy,
                                                                        const ablate::eos::tChemSoot::Temperature::real_type_2d_view_host_type& state,
                                                                        const ablate::eos::tChemSoot::Temperature::real_type_1d_view_host_type& internalEnergyRef,
                                                                        const Temperature::real_type_2d_view_host_type& enthalpyMass, const Temperature::real_type_1d_view_host_type& enthalpyReference,
                                                                        const ablate::eos::tChemSoot::Temperature::kinetic_model_host_type& kmcd) {
    ablate::eos::tChemSoot::impl::Temperature_TemplateRun("ablate::eos::tChemSoot::Temperature::runHostBatch", policy, state, internalEnergyRef, enthalpyMass, enthalpyReference, kmcd);
}
#endif