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
#ifndef ABLATELIBRARY_TCHEM_IGNITION_ZEROD_SOOT_HPP
#define ABLATELIBRARY_TCHEM_IGNITION_ZEROD_SOOT_HPP
#include <Kokkos_Macros.hpp>
#ifndef KOKKOS_ENABLE_CUDA

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

#include "IgnitionZeroDSoot_Implemenatation.hpp"
namespace ablate::eos::tChemSoot {

using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
using device_type = typename Tines::UseThisDevice<exec_space>::type;
using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;

struct IgnitionZeroDSoot {
    static inline ordinal_type getWorkSpaceSize(const KineticModelConstData<device_type>& kmcd) {
        using problem_type = IgnitionZeroD_ProblemSoot<real_type, device_type>;
        const ordinal_type m = problem_type::getNumberOfEquations(kmcd) + ordinal_type(1);
        ordinal_type work_size(0);
        work_size = IgnitionZeroDSootImplementation::getWorkSpaceSize(kmcd);

        return work_size + m;
    }

    /// tadv - an input structure for time marching
    /// state (nSpec+3) - initial condition of the state vector
    /// work - work space sized by getWorkSpaceSize
    /// t_out - time when this code exits
    /// state_out - final condition of the state vector (the same input state can
    /// be overwritten) kmcd - const data for kinetic model
    static void runDeviceBatch(  /// thread block size
        typename UseThisTeamPolicy<exec_space>::type& policy,
        /// global tolerence parameters that governs all samples
        const real_type_1d_view& tol_newton, const real_type_2d_view& tol_time,
        /// sample specific input
        const real_type_2d_view& fac, const time_advance_type_1d_view& tadv, const real_type_2d_view& state, const real_type_1d_view& HF, const real_type_2d_view_host& Hi_Scratch,
        const real_type_2d_view_host& cp_gas_scratch,
        /// output
        const real_type_1d_view& t_out, const real_type_1d_view& dt_out, const real_type_2d_view& state_out,
        /// const data from kinetic model
        const Tines::value_type_1d_view<KineticModelConstData<interf_device_type>, interf_device_type>& kmcds, double thresholdTemperature);

    static void runHostBatch(  /// input
        typename UseThisTeamPolicy<host_exec_space>::type& policy,
        /// global tolerence parameters that governs all samples
        const real_type_1d_view_host& tol_newton, const real_type_2d_view_host& tol_time,
        /// sample specific input
        const real_type_2d_view_host& fac, const time_advance_type_1d_view_host& tadv, const real_type_2d_view_host& state, const real_type_1d_view& HF, const real_type_2d_view_host& Hi_Scratch,
        const real_type_2d_view_host& cp_gas_scratch,
        /// output
        const real_type_1d_view_host& t_out, const real_type_1d_view_host& dt_out, const real_type_2d_view_host& state_out,
        /// const data from kinetic model
        const Tines::value_type_1d_view<KineticModelConstData<interf_host_device_type>, interf_host_device_type>& kmcds, double thresholdTemperature);
};

}  // namespace ablate::eos::tChemSoot

#endif
#endif
