/* =====================================================================================
TChem version 2.0
Copyright (2020) NTESS
https://github.com/sandialabs/TChem

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

This file is part of TChem. TChem is open source software: you can redistribute it
and/or modify it under the terms of BSD 2-Clause License
(https://opensource.org/licenses/BSD-2-Clause). A copy of the licese is also
provided under the main directory

Questions? Contact Cosmin Safta at <csafta@sandia.gov>, or
           Kyungjoo Kim at <kyukim@sandia.gov>, or
           Oscar Diaz-Ibarra at <odiazib@sandia.gov>

Sandia National Laboratories, Livermore, CA, USA
===================================================================================== */

#ifndef ABLATELIBRARY_TCHEM2_CONSTANTVOLUMEIGNITIONREACTORTEMPERATURETHRESHOLD_HPP
#define ABLATELIBRARY_TCHEM2_CONSTANTVOLUMEIGNITIONREACTORTEMPERATURETHRESHOLD_HPP

#include <TChem_ConstantVolumeIgnitionReactor.hpp>
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace ablate::eos::tChem {
template <typename PolicyType, typename ValueType, typename DeviceType>
void ConstantVolumeIgnitionReactor_TemplateRun(  /// required template arguments
    const std::string& profile_name, const ValueType& dummyValueType,
    /// team size setting
    const PolicyType& policy,
    /// control
    const bool solve_tla, const real_type theta_tla,
    /// input
    const Tines::value_type_1d_view<real_type, DeviceType>& tol_newton, const Tines::value_type_2d_view<real_type, DeviceType>& tol_time, const Tines::value_type_2d_view<real_type, DeviceType>& fac,
    const Tines::value_type_1d_view<time_advance_type, DeviceType>& tadv,
    //    /// state (nSample, nSpec+1)
    const Tines::value_type_2d_view<real_type, DeviceType>& state,
    //    /// state_z (nSample, nSpec, nReac)
    const Tines::value_type_3d_view<real_type, DeviceType>& state_z,
    //    /// output
    const Tines::value_type_1d_view<real_type, DeviceType>& t_out, const Tines::value_type_1d_view<real_type, DeviceType>& dt_out, const Tines::value_type_2d_view<real_type, DeviceType>& state_out,
    const Tines::value_type_3d_view<real_type, DeviceType>& state_z_out,
    //    /// const data from kinetic model
    const Tines::value_type_1d_view<KineticModelConstData<DeviceType>, DeviceType>& kmcds, double thresholdTemperature) {
    Kokkos::Profiling::pushRegion(profile_name);

    using policy_type = PolicyType;
    using value_type = ValueType;
    using device_type = DeviceType;

    using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;

    /// this is not necessary when we receive workspace explicitly
    auto kmcd_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(kmcds, 0));

    const ordinal_type level = 1;
    const ordinal_type per_team_extent = ConstantVolumeIgnitionReactor::getWorkSpaceSize(solve_tla, kmcd_host());

    Kokkos::parallel_for(
        profile_name, policy, KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const ordinal_type i = member.league_rank();
            const real_type zero(0);

            const auto kmcd_at_i = (kmcds.extent(0) == 1 ? kmcds(0) : kmcds(i));
            const auto tadv_at_i = tadv(i);
            const real_type t_end = tadv_at_i._tend;
            const real_type_0d_view_type t_out_at_i = Kokkos::subview(t_out, i);
            if (t_out_at_i() < t_end) {
                const real_type_1d_view_type state_at_i = Kokkos::subview(state, i, Kokkos::ALL());
                const real_type_1d_view_type state_out_at_i = Kokkos::subview(state_out, i, Kokkos::ALL());
                const real_type_1d_view_type fac_at_i = Kokkos::subview(fac, i, Kokkos::ALL());

                const real_type_0d_view_type dt_out_at_i = Kokkos::subview(dt_out, i);
                Scratch<real_type_1d_view_type> work(member.team_scratch(level), per_team_extent);

                Impl::StateVector<real_type_1d_view_type> sv_at_i(kmcd_at_i.nSpec, state_at_i);
                Impl::StateVector<real_type_1d_view_type> sv_out_at_i(kmcd_at_i.nSpec, state_out_at_i);
                TCHEM_CHECK_ERROR(!sv_at_i.isValid(), "Error: input state vector is not valid");
                TCHEM_CHECK_ERROR(!sv_out_at_i.isValid(), "Error: input state vector is not valid");
                {
                    const ordinal_type jacobian_interval = tadv_at_i._jacobian_interval;
                    const ordinal_type max_num_newton_iterations = tadv_at_i._max_num_newton_iterations;
                    const ordinal_type max_num_time_iterations = tadv_at_i._num_time_iterations_per_interval;
                    const ordinal_type max_num_outer_time_iterations = tadv_at_i._num_outer_time_iterations_per_interval;

                    const real_type dt_in = tadv_at_i._dt, dt_min = tadv_at_i._dtmin, dt_max = tadv_at_i._dtmax;
                    const real_type t_beg = tadv_at_i._tbeg;

                    const real_type temperature = sv_at_i.Temperature();

                    // only bother to integrate this point in the ode if the temperature is greater than thresholdTemperature
                    if (temperature > thresholdTemperature) {
                        const real_type density = sv_at_i.Density();
                        const real_type_1d_view_type Ys = sv_at_i.MassFractions();

                        const real_type_0d_view_type temperature_out(sv_out_at_i.TemperaturePtr());
                        const real_type_0d_view_type pressure_out(sv_out_at_i.PressurePtr());
                        const real_type_1d_view_type Ys_out = sv_out_at_i.MassFractions();
                        const real_type_0d_view_type density_out(sv_out_at_i.DensityPtr());

                        const ordinal_type m = kmcd_at_i.nSpec + 1;
                        auto wptr = work.data();
                        const real_type_1d_view_type vals(wptr, m);
                        wptr += m;
                        const real_type_1d_view_type vals_out(wptr, m);
                        wptr += m;  /// added m workspace
                        const real_type_1d_view_type ww(wptr, work.extent(0) - (wptr - work.data()));

                        /// we can only guarantee vals is contiguous array. we basically assume
                        /// that a state vector can be arbitrary ordered.

                        /// m is nSpec + 1
                        ::TChem::ConstantVolumeIgnitionReactor ::packToValues(member, temperature, Ys, vals);
                        member.team_barrier();

                        using constant_volume_ignition_type = Impl::ConstantVolumeIgnitionReactor<value_type, device_type>;

                        real_type t_cur(t_out_at_i());
                        for (ordinal_type iter = 0; iter < max_num_outer_time_iterations; ++iter) {
                            /// advance time of the reactor
                            constant_volume_ignition_type ::team_invoke(member,
                                                                        jacobian_interval,
                                                                        max_num_newton_iterations,
                                                                        max_num_time_iterations,
                                                                        tol_newton,
                                                                        tol_time,
                                                                        fac_at_i,
                                                                        dt_in,
                                                                        dt_min,
                                                                        dt_max,
                                                                        t_beg,
                                                                        t_end,
                                                                        density,
                                                                        vals,
                                                                        t_out_at_i,
                                                                        dt_out_at_i,
                                                                        vals_out,
                                                                        ww,
                                                                        kmcd_at_i);

                            /// if tla is requested and the reactor is advanced
                            const real_type dt_tla = t_out_at_i() - t_cur;
                            if (solve_tla && dt_tla > zero) {
                                auto wtmp = ww.data();
                                const ordinal_type n = kmcd_at_i.nReac;

                                /// temperature
                                const real_type temperature_a = vals(0), temperature_b = vals_out(0);

                                /// problem type computes Jacobian
                                using problem_type = Impl::ConstantVolumeIgnitionReactor_Problem<value_type, device_type>;

                                /// compute J
                                real_type_2d_view_type J_a(wtmp, m, m);
                                wtmp += J_a.span();
                                real_type_2d_view_type J_b(wtmp, m, m);
                                wtmp += J_b.span();
                                {
                                    problem_type problem;
                                    problem._density = density;
                                    problem._fac = fac_at_i;
                                    problem._kmcd = kmcd_at_i;

                                    real_type_1d_view_type pw(wtmp, problem.getWorkSpaceSize());
                                    problem._work = pw;

                                    problem.computeJacobian(member, vals, J_a);
                                    problem.computeJacobian(member, vals_out, J_b);
                                }

                                /// compute pressure (pressure is a dependent variable)
                                real_type pressure_computed_a(0), pressure_computed_b(0);
                                const auto Ys_vals_a = Kokkos::subview(vals, Kokkos::make_pair(1, m));
                                const auto Ys_vals_b = Kokkos::subview(vals_out, Kokkos::make_pair(1, m));
                                {
                                    using molar_weights = Impl::MolarWeights<real_type, device_type>;
                                    const real_type Wmix_a = molar_weights::team_invoke(member, Ys_vals_a, kmcd_at_i);
                                    const real_type Wmix_b = molar_weights::team_invoke(member, Ys_vals_b, kmcd_at_i);
                                    member.team_barrier();
                                    pressure_computed_a = density * kmcd_at_i.Runiv * temperature_a / Wmix_a;
                                    pressure_computed_b = density * kmcd_at_i.Runiv * temperature_b / Wmix_b;
                                }

                                /// advance tla system
                                {
                                    using tla_type = Impl::TangentLinearApproximationIgnitionDelayTime<real_type, device_type, Impl::ConstantVolumeIgnitionReactorTangetLinearApproximationSourceTerm>;

                                    const ordinal_type wsize_tla = tla_type::getWorkSpaceSize(kmcd_at_i);
                                    auto state_z_at_i = Kokkos::subview(state_z, i, Kokkos::ALL(), Kokkos::ALL());
                                    auto state_z_out_at_i = Kokkos::subview(state_z_out, i, Kokkos::ALL(), Kokkos::ALL());
                                    real_type_1d_view_type work_tla(wtmp, wsize_tla);
                                    tla_type ::team_invoke(member,
                                                           theta_tla,
                                                           dt_tla,
                                                           density,
                                                           pressure_computed_a,
                                                           pressure_computed_b,
                                                           temperature_a,
                                                           temperature_b,
                                                           Ys_vals_a,
                                                           Ys_vals_b,
                                                           J_a,
                                                           J_b,
                                                           state_z_at_i,
                                                           state_z_out_at_i,
                                                           work_tla,
                                                           kmcd_at_i);

                                    /// if state vectors (in/out) are different
                                    if (state_z.data() != state_z_out.data()) {
                                        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m * n), [=](const ordinal_type& kk) {
                                            const ordinal_type k0 = kk / n, k1 = kk % n;
                                            state_z_at_i(k0, k1) = state_z_out_at_i(k0, k1);
                                        });
                                        member.team_barrier();
                                    }
                                }
                            }

                            /// if state vectors (in/out) are different
                            if (vals.data() != vals_out.data()) {
                                Kokkos::parallel_for(Kokkos::TeamVectorRange(member, m), [=](const ordinal_type& i) { vals(i) = vals_out(i); });
                                member.team_barrier();
                            }

                            /// update the current time
                            t_cur = t_out_at_i();
                        }

                        member.team_barrier();
                        ::TChem::ConstantVolumeIgnitionReactor ::unpackFromValues(member, vals_out, temperature_out, Ys_out);
                        member.team_barrier();

                        // update density and pressure with out data
                        {
                            const real_type Wmix = Impl::MolarWeights<real_type, device_type>::team_invoke(member, Ys_out, kmcd_at_i);

                            Kokkos::single(Kokkos::PerTeam(member), [&]() {
                                density_out() = density;  // density is constant
                                pressure_out() = density * kmcd_at_i.Runiv * temperature_out() / Wmix;
                            });
                            member.team_barrier();
                        }
                    } else {
                        for (ordinal_type s = 0; s < sv_at_i.size(); ++s) {
                            state_out_at_i[s] = state_at_i[s];
                        }
                    }
                }
            }
        });
    Kokkos::Profiling::popRegion();
}

struct ConstantVolumeIgnitionReactorTemperatureThreshold {
   public:
    /// tadv - an input structure for time marching
    /// state (nSpec+3) - initial condition of the state vector
    /// work - work space sized by getWorkSpaceSize
    /// t_out - time when this code exits
    /// state_out - final condition of the state vector (the same input state can
    /// be overwritten) kmcd - const data for kinetic model
    static void runDeviceBatch(  /// thread block size
        typename UseThisTeamPolicy<exec_space>::type& policy, const bool& solve_tla, const real_type& theta_tla,
        /// input
        const real_type_1d_view& tol_newton, const real_type_2d_view& tol_time, const real_type_2d_view& fac, const time_advance_type_1d_view& tadv, const real_type_2d_view& state,
        const real_type_3d_view& state_z,
        /// output
        const real_type_1d_view& t_out, const real_type_1d_view& dt_out, const real_type_2d_view& state_out, const real_type_3d_view& state_z_out,
        /// const data from kinetic model
        const Tines::value_type_1d_view<KineticModelConstData<interf_device_type>, interf_device_type>& kmcds, double thresholdTemperature) {
        const std::string profile_name = "ablate::eos::tChem::IgnitionZeroDTemperatureThreshold::runDeviceBatch::kmcd array";
        using value_type = real_type;

        ConstantVolumeIgnitionReactor_TemplateRun(
            profile_name, value_type(), policy, solve_tla, theta_tla, tol_newton, tol_time, fac, tadv, state, state_z, t_out, dt_out, state_out, state_z_out, kmcds, thresholdTemperature);
    }
};

}  // namespace ablate::eos::tChem
#endif
