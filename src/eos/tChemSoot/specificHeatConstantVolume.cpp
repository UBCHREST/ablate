#include "specificHeatConstantVolume.hpp"
#include "TChem_Impl_CpMixMs.hpp"
#include "TChem_Impl_MolarWeights.hpp"
#include "eos/tChemSoot.hpp"

namespace tChemLib = TChem;
namespace ablate::eos::tChemSoot::impl {

template <typename PolicyType, typename DeviceType>
void SpecificHeatConstantVolume_TemplateRun(const std::string& profile_name,
                                            /// team size setting
                                            const PolicyType& policy, const Tines::value_type_2d_view<real_type, DeviceType>& state, const Tines::value_type_1d_view<real_type, DeviceType>& CvMix,
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
            const StateVectorSoot<real_type_1d_view_type> sv_at_i_total(kmcd.nSpec, state_at_i);

            // Get the Gaseous State Vector
            real_type_1d_view_type state_at_i_gas = real_type_1d_view_type("Gaseous", ::TChem::Impl::getStateVectorSize(kmcd.nSpec));
            Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec, state_at_i_gas);
            // Need to scale Yi appropriately
            sv_at_i_total.SplitYiState(sv_at_i);

            const real_type_0d_view_type CvMix_at_i = Kokkos::subview(CvMix, i);

            Scratch<real_type_1d_view_type> work(member.team_scratch(level), per_team_extent);
            auto cpks = real_type_1d_view_type((real_type*)work.data(), kmcd.nSpec);

            TCHEM_CHECK_ERROR(!sv_at_i.isValid(), "Error: input state vector is not valid");
            {
                real_type t = sv_at_i.Temperature();
                const real_type_1d_view_type ys = sv_at_i.MassFractions();
                const real_type Yc = sv_at_i_total.MassFractionCarbon();
                // compute cv_gas
                real_type wmix = tChemLib::Impl::MolarWeights<real_type, device_type>::team_invoke(member, ys, kmcd);
                member.team_barrier();
                auto cv_gas = tChemLib::Impl::CpMixMs<real_type, device_type>::team_invoke(member, t, ys, cpks, kmcd) - kmcd.Runiv / wmix;
                auto cv_Soot = (ablate::eos::TChemSoot::CarbonCp_R(t) - 1) * kmcd.Runiv / ablate::eos::tChemSoot::MWCarbon;
                CvMix_at_i() = (1 - Yc) * cv_gas + Yc * cv_Soot;
            }
        });
    Kokkos::Profiling::popRegion();
}

}  // namespace ablate::eos::tChemSoot::impl

[[maybe_unused]] void ablate::eos::tChemSoot::SpecificHeatConstantVolume::runDeviceBatch(typename UseThisTeamPolicy<exec_space>::type& policy,
                                                                                         const ablate::eos::tChemSoot::SpecificHeatConstantVolume::real_type_2d_view_type& state,
                                                                                         const ablate::eos::tChemSoot::SpecificHeatConstantVolume::real_type_1d_view_type& CvMix,
                                                                                         const ablate::eos::tChemSoot::SpecificHeatConstantVolume::kinetic_model_type& kmcd) {
    ablate::eos::tChemSoot::impl::SpecificHeatConstantVolume_TemplateRun("ablate::eos::tChemSoot::SpecificHeatConstantVolume::runDeviceBatch", policy, state, CvMix, kmcd);
}

[[maybe_unused]] void ablate::eos::tChemSoot::SpecificHeatConstantVolume::runHostBatch(typename UseThisTeamPolicy<host_exec_space>::type& policy,
                                                                                       const ablate::eos::tChemSoot::SpecificHeatConstantVolume::real_type_2d_view_host_type& state,
                                                                                       const ablate::eos::tChemSoot::SpecificHeatConstantVolume::real_type_1d_view_host_type& CvMix,
                                                                                       const ablate::eos::tChemSoot::SpecificHeatConstantVolume::kinetic_model_host_type& kmcd) {
    ablate::eos::tChemSoot::impl::SpecificHeatConstantVolume_TemplateRun("ablate::eos::tChemSoot::SpecificHeatConstantVolume::runHostBatch", policy, state, CvMix, kmcd);
}
