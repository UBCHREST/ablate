#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_Impl_MolarWeights.hpp"
#include "speedOfSound.hpp"

namespace tChemLib = TChem;

namespace ablate::eos::tChem::impl {
template <typename PolicyType, typename DeviceType>
void SpeedOfSound_TemplateRun(const std::string& profile_name,
                              /// team size setting
                              const PolicyType& policy, const Tines::value_type_2d_view<real_type, DeviceType>& state, const Tines::value_type_1d_view<real_type, DeviceType>& speedOfSound,
                              const KineticModelConstData<DeviceType>& kmcd) {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;

    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;

    const ordinal_type level = 1;
    const ordinal_type per_team_extent = SpeedOfSound::getWorkSpaceSize(kmcd.nSpec);

    Kokkos::parallel_for(
        profile_name, policy, KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const ordinal_type i = member.league_rank();
            const real_type_1d_view_type state_at_i = Kokkos::subview(state, i, Kokkos::ALL());
            const real_type_0d_view_type speedOfSound_at_i = Kokkos::subview(speedOfSound, i);

            Scratch<real_type_1d_view_type> work(member.team_scratch(level), per_team_extent);
            auto cpks = real_type_1d_view_type((real_type*)work.data(), kmcd.nSpec);

            const Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec, state_at_i);
            TCHEM_CHECK_ERROR(!sv_at_i.isValid(), "Error: input state vector is not valid");
            {
                const real_type t = sv_at_i.Temperature();
                const real_type_1d_view_type ys = sv_at_i.MassFractions();

                // compute cp
                auto cp = tChemLib::Impl::CpMixMs<real_type, device_type>::team_invoke(member, t, ys, cpks, kmcd);

                // compute the mw of the mix
                const real_type mwMix = tChemLib::Impl::MolarWeights<real_type, device_type>::team_invoke(member, sv_at_i.MassFractions(), kmcd);
                member.team_barrier();

                // compute gamma
                auto R = kmcd.Runiv / mwMix;
                auto cv = cp - R;
                auto gamma = cp / cv;
                speedOfSound_at_i() = Tines::ats<real_type>::sqrt(gamma * R * t);
            }
        });
    Kokkos::Profiling::popRegion();
}

}  // namespace ablate::eos::tChem::impl

void ablate::eos::tChem::SpeedOfSound::runDeviceBatch(typename UseThisTeamPolicy<exec_space>::type& policy, const SpeedOfSound::real_type_2d_view_type& state,
                                                          const SpeedOfSound::real_type_1d_view_type& speedOfSound,
                                                          const SpeedOfSound::kinetic_model_type& kmcd) {
    ablate::eos::tChem::impl::SpeedOfSound_TemplateRun("ablate::eos::tChem::SensibleEnthalpy::runDeviceBatch", policy, state, speedOfSound, kmcd);
}
void ablate::eos::tChem::SpeedOfSound::runHostBatch(typename UseThisTeamPolicy<host_exec_space>::type& policy, const ablate::eos::tChem::SpeedOfSound::real_type_2d_view_host_type& state,
                                                        const ablate::eos::tChem::SpeedOfSound::real_type_1d_view_host_type& speedOfSound,
                                                        const ablate::eos::tChem::SpeedOfSound::kinetic_model_type& kmcd) {
    ablate::eos::tChem::impl::SpeedOfSound_TemplateRun("ablate::eos::tChem::SensibleEnthalpy::runHostBatch", policy, state, speedOfSound, kmcd);
}
