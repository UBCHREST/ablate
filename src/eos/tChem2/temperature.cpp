#include "temperature.hpp"
#include "eos/tChem/sensibleInternalEnergyFcn.hpp"

namespace ablate::eos::tChem2::impl {
template <typename PolicyType, typename DeviceType>
void Temperature_TemplateRun(const std::string& profile_name,
                             /// team size setting
                             const PolicyType& policy, const Tines::value_type_2d_view<real_type, DeviceType>& state, const Tines::value_type_1d_view<real_type, DeviceType>& internalEnergyRef,
                             const Tines::value_type_2d_view<real_type, DeviceType>& enthalpyMass, const Tines::value_type_1d_view<real_type, DeviceType>& enthalpyReference,
                             const KineticModelConstData<DeviceType>& kmcd) {
    Kokkos::Profiling::pushRegion(profile_name);
    using policy_type = PolicyType;
    using device_type = DeviceType;

    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;

    const ordinal_type level = 1;
    const ordinal_type per_team_extent = Temperature::getWorkSpaceSize(kmcd.nSpec);

    Kokkos::parallel_for(
        profile_name, policy, KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const ordinal_type i = member.league_rank();
            const real_type_1d_view_type state_at_i = Kokkos::subview(state, i, Kokkos::ALL());
            const real_type_1d_view_type hi_at_i = Kokkos::subview(enthalpyMass, i, Kokkos::ALL());
            const real_type_0d_view_type internalEnergyRef_at_i = Kokkos::subview(internalEnergyRef, i);

            Scratch<real_type_1d_view_type> work(member.team_scratch(level), per_team_extent);
            auto cpks = real_type_1d_view_type((real_type*)work.data(), kmcd.nSpec);

            const Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec, state_at_i);
            TCHEM_CHECK_ERROR(!sv_at_i.isValid(), "Error: input state vector is not valid");
            {
                real_type& t = sv_at_i.Temperature();
                double t2 = t;
                const real_type_1d_view_type ys = sv_at_i.MassFractions();

                // set some constants
                const auto EPS_T_RHO_E = 1E-8;
                const auto ITERMAX_T = 100;

                // compute the first error
                double e2 = ablate::eos::tChem::impl::SensibleInternalEnergyFcn<real_type, device_type>::team_invoke(member, t, ys, hi_at_i, cpks, enthalpyReference, kmcd);
                double f2 = internalEnergyRef_at_i() - e2;
                if (Kokkos::abs(f2) > EPS_T_RHO_E) {
                    double t0 = t2;
                    double f0 = f2;
                    double t1 = t0 + 1;

                    t = t1;
                    double e1 = ablate::eos::tChem::impl::SensibleInternalEnergyFcn<real_type, device_type>::team_invoke(member, t, ys, hi_at_i, cpks, enthalpyReference, kmcd);
                    double f1 = internalEnergyRef_at_i() - e1;

                    for (int it = 0; it < ITERMAX_T; it++) {
                        t2 = t1 - f1 * (t1 - t0) / (f1 - f0 + 1E-30);
                        t2 = Kokkos::max(1.0, t2);
                        t = t2;
                        e2 = ablate::eos::tChem::impl::SensibleInternalEnergyFcn<real_type, device_type>::team_invoke(member, t, ys, hi_at_i, cpks, enthalpyReference, kmcd);
                        f2 = internalEnergyRef_at_i() - e2;
                        if (Tines::ats<real_type>::abs(f2) <= EPS_T_RHO_E) {
                            t = t2;
                            return;
                        }
                        t0 = t1;
                        t1 = t2;
                        f0 = f1;
                        f1 = f2;
                    }

                    // We iterated through the possible iterations, try again with a new guess
                    t0 = 1000.;
                    f0 = internalEnergyRef_at_i() - ablate::eos::tChem::impl::SensibleInternalEnergyFcn<real_type, device_type>::team_invoke(member, t, ys, hi_at_i, cpks, enthalpyReference, kmcd);
                    t1 = t0 + 1;
                    t = t1;
                    e1 = ablate::eos::tChem::impl::SensibleInternalEnergyFcn<real_type, device_type>::team_invoke(member, t, ys, hi_at_i, cpks, enthalpyReference, kmcd);
                    f1 = internalEnergyRef_at_i() - e1;

                    for (int it = 0; it < ITERMAX_T; it++) {
                        t2 = t1 - f1 * (t1 - t0) / (f1 - f0 + 1E-30);
                        t2 = Kokkos::max(1.0, t2);
                        t = t2;
                        e2 = ablate::eos::tChem::impl::SensibleInternalEnergyFcn<real_type, device_type>::team_invoke(member, t, ys, hi_at_i, cpks, enthalpyReference, kmcd);
                        f2 = internalEnergyRef_at_i() - e2;
                        if (Tines::ats<real_type>::abs(f2) <= EPS_T_RHO_E) {
                            t = t2;
                            return;
                        }
                        t0 = t1;
                        t1 = t2;
                        f0 = f1;
                        f1 = f2;
                    }

                    t = t2;
                }
            }
        });
    Kokkos::Profiling::popRegion();
}

}  // namespace ablate::eos::tChem::impl

[[maybe_unused]] void ablate::eos::tChem2::Temperature::runDeviceBatch(typename UseThisTeamPolicy<exec_space>::type& policy, const Temperature::real_type_2d_view_type& state,
                                                                      const Temperature::real_type_1d_view_type& internalEnergyRef, const Temperature::real_type_2d_view_type& enthalpyMass,
                                                                      const Temperature::real_type_1d_view_type& enthalpyReference, const Temperature::kinetic_model_type& kmcd) {
    ablate::eos::tChem2::impl::Temperature_TemplateRun("ablate::eos::tChem::Temperature::runDeviceBatch", policy, state, internalEnergyRef, enthalpyMass, enthalpyReference, kmcd);
}

[[maybe_unused]] void ablate::eos::tChem2::Temperature::runHostBatch(const typename UseThisTeamPolicy<host_exec_space>::type& policy,
                                                                    const ablate::eos::tChem2::Temperature::real_type_2d_view_host_type& state,
                                                                    const ablate::eos::tChem2::Temperature::real_type_1d_view_host_type& internalEnergyRef,
                                                                    const Temperature::real_type_2d_view_host_type& enthalpyMass, const Temperature::real_type_1d_view_host_type& enthalpyReference,
                                                                    const ablate::eos::tChem2::Temperature::kinetic_model_host_type& kmcd) {
    ablate::eos::tChem2::impl::Temperature_TemplateRun("ablate::eos::tChem::Temperature::runHostBatch", policy, state, internalEnergyRef, enthalpyMass, enthalpyReference, kmcd);
}
