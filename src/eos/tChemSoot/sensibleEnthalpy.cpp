#include "sensibleEnthalpy.hpp"
#include "eos/tChem/sensibleEnthalpyFcn.hpp"
#include "eos/tChemSoot.hpp"
#include "sensibleInternalEnergy.hpp"

namespace ablate::eos::tChemSoot::impl {
// For A LHF Approximation with both gaseous and solid species, H_{sens} = sum_{all spec} Y_{tot,i}H_{sens,i} = (1-Y_{soot}) H_{sens,gas} + Y_{soot} H_{sens,soot};
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
            const real_type_1d_view_type state_at_i = Kokkos::subview(state, i, Kokkos::ALL());      // State at current point of interest
            const real_type_0d_view_type& hMix_at_i = Kokkos::subview(enthalpyMassMixture, i);       // Output Location
            const real_type_1d_view_type hi_at_i = Kokkos::subview(enthalpyMass, i, Kokkos::ALL());  // Scratch Variable of h_species at point of interest
            const real_type_1d_view_type hi_at_i_gas = Kokkos::subview(hi_at_i, std::make_pair(1, kmcd.nSpec));
            const StateVectorSoot<real_type_1d_view_type> sv_at_i_total(kmcd.nSpec, state_at_i);

            // Want to reUse sensible enthalpy function from Non Soot case, meaning we need to split our state vector into a gaseous statevector
            real_type_1d_view_type state_at_i_gas = real_type_1d_view_type("Gaseous", Impl::getStateVectorSize(kmcd.nSpec));
            // Get the Gaseous State Vector
            Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec, state_at_i_gas);

            sv_at_i_total.SplitYiState(sv_at_i);

            Scratch<real_type_1d_view_type> work(member.team_scratch(level), per_team_extent);
            auto cpks = real_type_1d_view_type((real_type*)work.data(), kmcd.nSpec);

            TCHEM_CHECK_ERROR(!sv_at_i.isValid(), "Error: input state vector is not valid");
            {
                const real_type t = sv_at_i.Temperature();
                const real_type_1d_view_type ys = sv_at_i.MassFractions();
                const real_type Yc = state_at_i(kmcd.nSpec + 3);
                // compute on this team thread
                const real_type_1d_view_type enthalpyRef_Gas = Kokkos::subview(enthalpyRef, std::make_pair(1, kmcd.nSpec));
                hMix_at_i() = ablate::eos::tChem::impl::SensibleEnthalpyFcn<real_type, device_type>::team_invoke(member, t, ys, hi_at_i_gas, cpks, enthalpyRef_Gas, kmcd);
                // Now scale gaseous enthalpy by amount of soot present in system
                hMix_at_i() *= (1 - Yc);
                // Calculate Solid Enthalpy and add it
                hi_at_i(0) = ablate::eos::TChemSoot::CarbonEnthalpy_R_T(t) * t * kmcd.Runiv / tChemSoot::MWCarbon - enthalpyRef(0);
                hMix_at_i() += hi_at_i(0) * Yc;
            }
        });
    Kokkos::Profiling::popRegion();
}

}  // namespace ablate::eos::tChemSoot::impl

[[maybe_unused]] void ablate::eos::tChemSoot::SensibleEnthalpy::runDeviceBatch(typename UseThisTeamPolicy<exec_space>::type& policy, const SensibleInternalEnergy::real_type_2d_view_type& state,
                                                                               const SensibleInternalEnergy::real_type_1d_view_type& enthalpyMassMixture,
                                                                               const SensibleInternalEnergy::real_type_2d_view_type& enthalpyMass,
                                                                               const SensibleInternalEnergy::real_type_1d_view_type& enthalpyRef,
                                                                               const SensibleInternalEnergy::kinetic_model_type& kmcd) {
    ablate::eos::tChemSoot::impl::SensibleEnthalpy_TemplateRun("ablate::eos::tChemSoot::SensibleEnthalpy::runDeviceBatch", policy, state, enthalpyMassMixture, enthalpyMass, enthalpyRef, kmcd);
}

[[maybe_unused]] void ablate::eos::tChemSoot::SensibleEnthalpy::runHostBatch(typename UseThisTeamPolicy<host_exec_space>::type& policy,
                                                                             const ablate::eos::tChem::SensibleInternalEnergy::real_type_2d_view_host_type& state,
                                                                             const ablate::eos::tChem::SensibleInternalEnergy::real_type_1d_view_host_type& enthalpyMassMixture,
                                                                             const ablate::eos::tChem::SensibleInternalEnergy::real_type_2d_view_host_type& enthalpyMass,
                                                                             const ablate::eos::tChem::SensibleInternalEnergy::real_type_1d_view_host_type& enthalpyRef,
                                                                             const ablate::eos::tChem::SensibleInternalEnergy::kinetic_model_type& kmcd) {
    ablate::eos::tChemSoot::impl::SensibleEnthalpy_TemplateRun("ablate::eos::tChemSoot::SensibleEnthalpy::runHostBatch", policy, state, enthalpyMassMixture, enthalpyMass, enthalpyRef, kmcd);
}
