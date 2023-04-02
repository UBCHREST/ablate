#include <Kokkos_Core.hpp>
#ifndef KOKKOS_ENABLE_CUDA

#include "sensibleInternalEnergy.hpp"
#include "sensibleInternalEnergyFcn.hpp"

namespace ablate::eos::tChemSoot::impl {
template <typename PolicyType, typename DeviceType>
void SensibleInternalEnergy_TemplateRun(const std::string& profile_name,
                                        /// team size setting
                                        const PolicyType& policy, const Tines::value_type_2d_view<real_type, DeviceType>& state,  // Full State
                                        const Tines::value_type_1d_view<real_type, DeviceType>& internalEnergy,                   // Internal EnergyMixture
                                        const Tines::value_type_2d_view<real_type, DeviceType>& enthalpyMass,                     // Enthalpy for all species Scratch
                                        const Tines::value_type_1d_view<real_type, DeviceType>& enthalpyRef,                      // Reference Enthalpies for all species
                                        const KineticModelConstData<DeviceType>& kmcd) {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;

    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;

    const ordinal_type level = 1;
    const ordinal_type per_team_extent = SensibleInternalEnergy::getWorkSpaceSize(kmcd.nSpec);

    Kokkos::parallel_for(
        profile_name, policy, KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const ordinal_type i = member.league_rank();
            const real_type_1d_view_type state_at_i = Kokkos::subview(state, i, Kokkos::ALL());
            const real_type_0d_view_type& internalEnergy_at_i = Kokkos::subview(internalEnergy, i);
            const real_type_1d_view_type hi_at_i = Kokkos::subview(enthalpyMass, i, Kokkos::ALL());

            const StateVectorSoot<real_type_1d_view_type> sv_at_i_total(kmcd.nSpec, state_at_i);

            Scratch<real_type_1d_view_type> work(member.team_scratch(level), per_team_extent);
            auto cpks = real_type_1d_view_type((real_type*)work.data(), kmcd.nSpec);

            // Need to scale Yi appropriately
            real_type_1d_view_type state_at_i_gas = real_type_1d_view_type("Gaseous", ::TChem::Impl::getStateVectorSize(kmcd.nSpec));
            // Get the Gaseous State Vector
            Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec, state_at_i_gas);
            sv_at_i_total.SplitYiState(sv_at_i);
            TCHEM_CHECK_ERROR(!sv_at_i.isValid(), "Error: input state vector is not valid");
            {
                const real_type Yc = state_at_i(kmcd.nSpec + 3);
                // compute on this team thread
                internalEnergy_at_i() = ablate::eos::tChemSoot::impl::SensibleInternalEnergyFcn<real_type, device_type>::team_invoke(member, Yc, sv_at_i, hi_at_i, cpks, enthalpyRef, kmcd);
            }
        });
    Kokkos::Profiling::popRegion();
}

}  // namespace ablate::eos::tChemSoot::impl

[[maybe_unused]] void ablate::eos::tChemSoot::SensibleInternalEnergy::runDeviceBatch(typename UseThisTeamPolicy<exec_space>::type& policy, const SensibleInternalEnergy::real_type_2d_view_type& state,
                                                                                     const SensibleInternalEnergy::real_type_1d_view_type& internalEnergyRef,
                                                                                     const SensibleInternalEnergy::real_type_2d_view_type& enthalpyMass,
                                                                                     const SensibleInternalEnergy::real_type_1d_view_type& enthalpyRef,
                                                                                     const SensibleInternalEnergy::kinetic_model_type& kmcd) {
    ablate::eos::tChemSoot::impl::SensibleInternalEnergy_TemplateRun("ablate::eos::tChem::SensibleInternalEnergy::runDeviceBatch", policy, state, internalEnergyRef, enthalpyMass, enthalpyRef, kmcd);
}

[[maybe_unused]] void ablate::eos::tChemSoot::SensibleInternalEnergy::runHostBatch(typename UseThisTeamPolicy<host_exec_space>::type& policy,
                                                                                   const ablate::eos::tChemSoot::SensibleInternalEnergy::real_type_2d_view_host_type& state,
                                                                                   const ablate::eos::tChemSoot::SensibleInternalEnergy::real_type_1d_view_host_type& internalEnergyRef,
                                                                                   const ablate::eos::tChemSoot::SensibleInternalEnergy::real_type_2d_view_host_type& enthalpyMass,
                                                                                   const ablate::eos::tChemSoot::SensibleInternalEnergy::real_type_1d_view_host_type& enthalpyRef,
                                                                                   const ablate::eos::tChemSoot::SensibleInternalEnergy::kinetic_model_host_type& kmcd) {
    ablate::eos::tChemSoot::impl::SensibleInternalEnergy_TemplateRun("ablate::eos::tChem::SensibleInternalEnergy::runHostBatch", policy, state, internalEnergyRef, enthalpyMass, enthalpyRef, kmcd);
}
#endif