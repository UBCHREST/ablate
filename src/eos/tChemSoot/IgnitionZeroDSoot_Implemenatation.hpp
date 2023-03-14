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
#ifndef ABLATELIBRARY_TCHEM_IGNITION_ZEROD_SOOT_IMPLEMENTATION_HPP
#define ABLATELIBRARY_TCHEM_IGNITION_ZEROD_SOOT_IMPLEMENTATION_HPP

#include "IgnitionZeroD_ProblemSoot.hpp"
#include "TChem_Util.hpp"

#include "Sacado.hpp"
#include "Tines.hpp"

namespace ablate::eos::tChemSoot {

struct IgnitionZeroDSootImplementation {
    using device_type = typename Tines::UseThisDevice<exec_space>::type;
    using value_type = real_type;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;
    using kinetic_model_type = KineticModelConstData<device_type>;

    using time_integrator_type = Tines::TimeIntegratorTrBDF2<value_type, device_type>;

    using problem_type = IgnitionZeroD_ProblemSoot<value_type, device_type>;

    static inline ordinal_type getWorkSpaceSize(const kinetic_model_type& kmcd) {
        problem_type problem;
        problem._kmcd = kmcd;
        ordinal_type work_size_problem = problem.getWorkSpaceSize();
        ordinal_type m = problem.getNumberOfEquations();
        ordinal_type wlen(0);
        time_integrator_type::workspace(m, wlen);
        return wlen + work_size_problem;
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION static void team_invoke_detail(const MemberType& member,
                                                          /// input iteration and qoi index to store
                                                          const ordinal_type& jacobian_interval, const ordinal_type& max_num_newton_iterations, const ordinal_type& max_num_time_iterations,
                                                          const real_type_1d_view_type& tol_newton, const real_type_2d_view_type& tol_time, const real_type_1d_view_type& fac,
                                                          /// input time step and time range
                                                          const real_type& dt_in, const real_type& dt_min, const real_type& dt_max, const real_type& t_beg, const real_type& t_end,
                                                          /// input (initial condition)
                                                          const real_type& TotalDensity,                  /// Total Density
                                                          const real_type_1d_view_type& h_formation_ref,  /// reference heats of formation
                                                          const real_type_1d_view_type& hi_scratch,       /// scratch enthalpy vector
                                                          const real_type_1d_view_type& cp_gas_scratch,   /// Specific heats of the gas scratch variable
                                                          const real_type_1d_view_type& vals,             /// Temperature, mass fraction (kmcd.nSpec), Carbon Mass Fraction, Soot Number density
                                                          /// output (final output conditions)
                                                          const real_type_0d_view_type& t_out, const real_type_0d_view_type& dt_out, const real_type_0d_view_type& pressure_out,
                                                          const real_type_1d_view_type& vals_out,
                                                          /// workspace
                                                          const real_type_1d_view_type& work,
                                                          /// const input from kinetic model
                                                          const kinetic_model_type& kmcd) {
        // Remember values coming in are the total mass fraction values, and Things We'll need to scale sources are :
        //   a. The volume fraction of gas, i.e. (1-fv)
        //   b. The Soot Reaction adjusters...
        const ordinal_type problem_workspace_size = problem_type::getWorkSpaceSize(kmcd);  // stuff in this function might have to get changed
        problem_type problem;
        problem._kmcd = kmcd;

        /// problem workspace
        auto wptr = work.data();
        auto pw = real_type_1d_view_type(wptr, problem_workspace_size);
        wptr += problem_workspace_size;

        /// error check
        const ordinal_type workspace_used(wptr - work.data()), workspace_extent(work.extent(0));
        if (workspace_used > workspace_extent) {
            Kokkos::abort("Error Ignition ZeroD Soot Sacado : workspace used is larger than it is provided\n");
        }

        /// time integrator workspace
        auto tw = real_type_1d_view_type(wptr, workspace_extent - workspace_used);

        /// initialize problem
        problem._densityTot = TotalDensity;  // Total Density (Also assumed constant over time)
        problem._work = pw;                  // problem workspace array
        problem._kmcd = kmcd;                // kinetic model
        problem._fac = fac;                  // Currently Unclear what this is :
        problem._hi_ref = h_formation_ref;
        problem._hi_Scratch = hi_scratch;
        problem.cp_gas_Scratch = cp_gas_scratch;

        const ordinal_type r_val = time_integrator_type::invoke(
            member, problem, jacobian_interval, max_num_newton_iterations, max_num_time_iterations, tol_newton, tol_time, dt_in, dt_min, dt_max, t_beg, t_end, vals, t_out, dt_out, vals_out, tw);

        /// pressure is constant, make sure it in the next restarting iteration
        Kokkos::single(Kokkos::PerTeam(member), [=]() {
            const real_type zero(0);
            if (r_val == 0) {
                // If it is a success set value to one to signal to recalculate it later on
                pressure_out() = 1;
            } else {
                pressure_out() = zero;
            }
        });
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION static void team_invoke(const MemberType& member,
                                                   /// input iteration and qoi index to store
                                                   const ordinal_type& jacobian_interval, const ordinal_type& max_num_newton_iterations, const ordinal_type& max_num_time_iterations,
                                                   const real_type_1d_view_type& tol_newton, const real_type_2d_view_type& tol_time, const real_type_1d_view_type& fac,
                                                   /// input time step and time range
                                                   const real_type& dt_in, const real_type& dt_min, const real_type& dt_max, const real_type& t_beg, const real_type& t_end,
                                                   /// input (initial condition)
                                                   const real_type& TotalDensity,                   /// Total Density
                                                   const real_type_1d_view_type& HeatsOfFormation,  /// species heats of formation
                                                   const real_type_1d_view_type& hi_scratch,        /// Scratch Enthalpy Vector
                                                   const real_type_1d_view_type& cp_gas_scratch,    /// Scratch cp_gas vector
                                                   const real_type_1d_view_type& vals,              /// temperature, mass fractions,carbon mass fraction, Soot number density
                                                   /// output (final output conditions)
                                                   const real_type_0d_view_type& t_out, const real_type_0d_view_type& dt_out, const real_type_0d_view_type& pressure_out,
                                                   const real_type_1d_view_type& vals_out,
                                                   /// workspace
                                                   const real_type_1d_view_type& work,
                                                   /// const input from kinetic model
                                                   const kinetic_model_type& kmcd) {
        // const real_type atol_newton = 1e-10, rtol_newton = 1e-6, tol_time = 1e-4;
        team_invoke_detail(member,
                           jacobian_interval,
                           max_num_newton_iterations,
                           max_num_time_iterations,
                           tol_newton,
                           tol_time,
                           fac,
                           dt_in,
                           dt_min,
                           dt_max,
                           t_beg,
                           t_end,
                           TotalDensity,
                           HeatsOfFormation,
                           hi_scratch,
                           cp_gas_scratch,
                           vals,
                           t_out,
                           dt_out,
                           pressure_out,
                           vals_out,
                           work,
                           kmcd);
        member.team_barrier();
        /// input is valid and output is not valid then, send warning message
#if !defined(__CUDA_ARCH__)
        const real_type zero(0);
        if (dt_in > zero && dt_out() < zero) {
            Kokkos::single(Kokkos::PerTeam(member), [&]() { printf("Warning: IgnitionZeroD sample id(%d) failed\n", int(member.league_rank())); });
        }
#endif
    }
};

}  // namespace ablate::eos::tChemSoot

#endif
