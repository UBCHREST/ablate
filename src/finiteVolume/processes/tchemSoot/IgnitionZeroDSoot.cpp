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
//#include "finiteVolume/processes/tchemSoot/IgnitionZeroDSoot.hpp"
#include "IgnitionZeroDSoot.hpp"
#include "TChem_Util.hpp"
#include "IgnitionZeroDSoot_Internal.hpp"

/// tadv - an input structure for time marching
/// state (nSpec+3) - initial condition of the state vector
/// qidx (lt nSpec+1) - QoI indices to store in qoi output
/// work - work space sized by getWorkSpaceSize
/// tcnt - time counter
/// qoi (time + qidx.extent(0)) - QoI output
/// kmcd - const data for kinetic model

namespace ablate::finiteVolume::processes::tchemSoot {
    using device_type = typename Tines::UseThisDevice<exec_space>::type;
    using real_type_0d_view_type = Tines::value_type_0d_view<real_type,device_type>;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type,device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type,device_type>;
    using kinetic_model_type= KineticModelConstData<device_type>;


    void
    IgnitionZeroDSoot::runDeviceBatch( /// thread block size
        typename UseThisTeamPolicy<exec_space>::type& policy,
        /// input
        const real_type_1d_view& tol_newton,
        const real_type_2d_view& tol_time,
        const real_type_2d_view& fac,
        const time_advance_type_1d_view& tadv,
        const real_type_2d_view& state,
        const real_type_1d_view& HF,
        const real_type_2d_view& Hi_Scratch,
        const real_type_2d_view& cp_gas_scratch,
        /// output
        const real_type_1d_view& t_out,
        const real_type_1d_view& dt_out,
        const real_type_2d_view& state_out,
        /// const data from kinetic model
        const Tines::value_type_1d_view<KineticModelConstData<interf_device_type>,interf_device_type>& kmcds)
    {
        const std::string profile_name = "TChem::IgnitionZeroDSoot::runDeviceBatch::kmcd array";
        using value_type = real_type;
        IgnitionZeroDSoot_TemplateRun(
            profile_name,
            value_type(),
            policy,
            tol_newton,
            tol_time,
            fac,
            tadv,
            state,
            HF,
            Hi_Scratch,
            cp_gas_scratch,
            t_out,
            dt_out,
            state_out,
            kmcds);
    }

  void
  IgnitionZeroDSoot::runHostBatch( /// thread block size
                              typename UseThisTeamPolicy<host_exec_space>::type& policy,
                              /// input
                              const real_type_1d_view_host& tol_newton,
                              const real_type_2d_view_host& tol_time,
                              const real_type_2d_view_host& fac,
                              const time_advance_type_1d_view_host& tadv,
                              const real_type_2d_view_host& state,
                              const real_type_1d_view_host& HF,
                              const real_type_2d_view_host& Hi_Scratch,
                              const real_type_2d_view_host& cp_gas_scratch,
                              /// output
                              const real_type_1d_view_host& t_out,
                              const real_type_1d_view_host& dt_out,
                              const real_type_2d_view_host& state_out,
                              /// const data from kinetic model
			      const Tines::value_type_1d_view<KineticModelConstData<interf_host_device_type>,interf_host_device_type>& kmcds) {
    const std::string profile_name = "TChem::IgnitionZeroDSoot::runHostBatch::kmcd array";
    using value_type = real_type;
    IgnitionZeroDSoot_TemplateRun(
        profile_name,
        value_type(),
        policy,
        tol_newton,
        tol_time,
        fac,
        tadv,
        state,
        HF,
        Hi_Scratch,
        cp_gas_scratch,
        t_out,
        dt_out,
        state_out,
        kmcds);
  }

} // namespace TChem