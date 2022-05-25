#include "sensibleInternalEnergy.hpp"
#include "temperature.hpp"

namespace ablate::eos::tChem::impl {
template <typename PolicyType, typename DeviceType>
void SensibleInternalEnergy_TemplateRun(const std::string& profile_name,
                             /// team size setting
                             const PolicyType& policy, const Tines::value_type_2d_view<real_type, DeviceType>& state, const Tines::value_type_1d_view<real_type, DeviceType>& internalEnergyRef,
                             const Tines::value_type_1d_view<real_type, DeviceType>& mwMix,
                       const KineticModelConstData<DeviceType>& kmcd) {
//    Kokkos::Profiling::pushRegion(profile_name);
//    using policy_type = PolicyType;
//    using device_type = DeviceType;
//
//    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
//    using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;
//
//    Kokkos::parallel_for(
//        profile_name, policy, KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
//            const ordinal_type i = member.league_rank();
//            const real_type_1d_view_type state_at_i = Kokkos::subview(state, i, Kokkos::ALL());
//            const real_type_0d_view_type internalEnergyRef_at_i = Kokkos::subview(internalEnergyRef, i);
//            const real_type_0d_view_type mwMix_at_i = Kokkos::subview(mwMix, i);
//
//            const Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd.nSpec, state_at_i);
//            TCHEM_CHECK_ERROR(!sv_at_i.isValid(), "Error: input state vector is not valid");
//            {
//                const real_type& t = sv_at_i.Temperature();
//                const real_type_1d_view_type Ys = sv_at_i.MassFractions();
//
//                // set some constants
//                const auto EPS_T_RHO_E = 1E-8;
//                const auto ITERMAX_T = 100;
//
//                // compute the first error
//                double e2;
//                tempYiWorkingArray[0] = t2;
//
//                int err = ComputeSensibleInternalEnergyInternal(numSpec, tempYiWorkingArray, mwMix, e2);
//                TCCHKERRQ(err);
//                double f2 = internalEnergyRef - e2;
//                T = t2;  // set for first guess
//                if (PetscAbs(f2) > EPS_T_RHO_E) {
//                    double t0 = t2;
//                    double f0 = f2;
//                    double t1 = t0 + 1;
//                    double e1;
//                    tempYiWorkingArray[0] = t1;
//                    err = ComputeSensibleInternalEnergyInternal(numSpec, tempYiWorkingArray, mwMix, e1);
//                    TCCHKERRQ(err);
//                    double f1 = internalEnergyRef - e1;
//
//                    for (int it = 0; it < ITERMAX_T; it++) {
//                        t2 = t1 - f1 * (t1 - t0) / (f1 - f0 + 1E-30);
//                        t2 = PetscMax(1.0, t2);
//                        tempYiWorkingArray[0] = t2;
//                        err = ComputeSensibleInternalEnergyInternal(numSpec, tempYiWorkingArray, mwMix, e2);
//                        TCCHKERRQ(err);
//                        f2 = internalEnergyRef - e2;
//                        if (PetscAbs(f2) <= EPS_T_RHO_E) {
//                            T = t2;
//                            PetscFunctionReturn(0);
//                        }
//                        t0 = t1;
//                        t1 = t2;
//                        f0 = f1;
//                        f1 = f2;
//                    }
//                    T = t2;
//                }
//
//
//                CpMixMass_at_i() = CpMixMs::team_invoke(member, t, Ys, CpMass_at_i, kmcd);
//            }
//        });
//    Kokkos::Profiling::popRegion();
}

}  // namespace ablate::eos::tChem::impl

void ablate::eos::tChem::SensibleInternalEnergy::runDeviceBatch(typename UseThisTeamPolicy<exec_space>::type& policy, const SensibleInternalEnergy::real_type_2d_view_type& state,
                                                          const SensibleInternalEnergy::real_type_1d_view_type& internalEnergyRef, const SensibleInternalEnergy::real_type_1d_view_type& mwMix, const Temperature::kinetic_model_type& kmcd) {

}
void ablate::eos::tChem::SensibleInternalEnergy::runHostBatch(typename UseThisTeamPolicy<host_exec_space>::type& policy, const ablate::eos::tChem::SensibleInternalEnergy::real_type_2d_view_host_type& state,
                                                        const ablate::eos::tChem::SensibleInternalEnergy::real_type_1d_view_host_type& internalEnergyRef,
                                                        const ablate::eos::tChem::SensibleInternalEnergy::real_type_1d_view_host_type& mwMix,
                                                        const ablate::eos::tChem::SensibleInternalEnergy::kinetic_model_type& kmcd) {}
