#include "sensibleEnthalpy.hpp"
#include "sensibleEnthalpyFcn.hpp"

namespace ablate::eos::tChem::impl {
template <typename PolicyType, typename DeviceType>
void SensibleEnthalpy_TemplateRun(const std::string& profile_name,
                                  /// team size setting
                                  const PolicyType& policy, const Tines::value_type_2d_view<real_type, DeviceType>& state, const Tines::value_type_1d_view<real_type, DeviceType>& enthalpyMassMixture,
                                  const Tines::value_type_2d_view<real_type, DeviceType>& enthalpyMass, const Tines::value_type_1d_view<real_type, DeviceType>& enthalpyRef,
                                  const KineticModelConstData<DeviceType>& kmcd) {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;

    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;

    const ordinal_type level = 1;
    const ordinal_type per_team_extent = SensibleEnthalpy::getWorkSpaceSize(kmcd.nSpec);

    Kokkos::parallel_for(
        profile_name, policy, KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const ordinal_type i = member.league_rank();
            const real_type_1d_view_type state_at_i = Kokkos::subview(state, i, Kokkos::ALL());
            const real_type_0d_view_type& hMix_at_i = Kokkos::subview(enthalpyMassMixture, i);
            const real_type_1d_view_type hi_at_i = Kokkos::subview(enthalpyMass, i, Kokkos::ALL());

            Scratch<real_type_1d_view_type> work(member.team_scratch(level), per_team_extent);
            auto cpks = real_type_1d_view_type((real_type*)work.data(), kmcd.nSpec);

            const Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec, state_at_i);
            TCHEM_CHECK_ERROR(!sv_at_i.isValid(), "Error: input state vector is not valid");
            {
                const real_type t = sv_at_i.Temperature();
                const real_type_1d_view_type ys = sv_at_i.MassFractions();

                // compute on this team thread
                hMix_at_i() = ablate::eos::tChem::impl::SensibleEnthalpyFcn<real_type, device_type>::team_invoke(member, t, ys, hi_at_i, cpks, enthalpyRef, kmcd);
            }
        });
    Kokkos::Profiling::popRegion();
}

}  // namespace ablate::eos::tChem::impl

[[maybe_unused]] void ablate::eos::tChem::SensibleEnthalpy::runDeviceBatch(typename UseThisTeamPolicy<exec_space>::type& policy, const SensibleInternalEnergy::real_type_2d_view_type& state,
                                                                           const SensibleInternalEnergy::real_type_1d_view_type& enthalpyMassMixture,
                                                                           const SensibleInternalEnergy::real_type_2d_view_type& enthalpyMass,
                                                                           const SensibleInternalEnergy::real_type_1d_view_type& enthalpyRef, const SensibleInternalEnergy::kinetic_model_type& kmcd) {
    ablate::eos::tChem::impl::SensibleEnthalpy_TemplateRun("ablate::eos::tChem::SensibleEnthalpy::runDeviceBatch", policy, state, enthalpyMassMixture, enthalpyMass, enthalpyRef, kmcd);
}

[[maybe_unused]] void ablate::eos::tChem::SensibleEnthalpy::runHostBatch(typename UseThisTeamPolicy<host_exec_space>::type& policy,
                                                                         const ablate::eos::tChem::SensibleInternalEnergy::real_type_2d_view_host_type& state,
                                                                         const ablate::eos::tChem::SensibleInternalEnergy::real_type_1d_view_host_type& enthalpyMassMixture,
                                                                         const ablate::eos::tChem::SensibleInternalEnergy::real_type_2d_view_host_type& enthalpyMass,
                                                                         const ablate::eos::tChem::SensibleInternalEnergy::real_type_1d_view_host_type& enthalpyRef,
                                                                         const ablate::eos::tChem::SensibleInternalEnergy::kinetic_model_type& kmcd) {
    ablate::eos::tChem::impl::SensibleEnthalpy_TemplateRun("ablate::eos::tChem::SensibleEnthalpy::runHostBatch", policy, state, enthalpyMassMixture, enthalpyMass, enthalpyRef, kmcd);
}
