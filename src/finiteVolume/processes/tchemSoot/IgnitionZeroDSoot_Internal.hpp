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
#include "IgnitionZeroDSoot.hpp"
#include "IgnitionZeroDSoot_Implemenatation.hpp"
#include "TChem_Util.hpp"

/// tadv - an input structure for time marching
/// state (nSpec+5) - initial condition of the state vector
/// qidx (lt nSpec+1) - QoI indices to store in qoi output
/// work - work space sized by getWorkSpaceSize
/// tcnt - time counter
/// qoi (time + qidx.extent(0)) - QoI output
/// kmcd - const data for kinetic model

namespace ablate::finiteVolume::processes::tchemSoot {

    template <typename PolicyType, typename DeviceType>
    void
    IgnitionZeroDSoot_TemplateRun( /// required template arguments
			    const std::string& profile_name,
                            const real_type& dummyValueType,
                            /// team size setting
                            const  PolicyType& policy,

                            /// input
                            const Tines::value_type_1d_view<real_type, DeviceType>& tol_newton,
                            const Tines::value_type_2d_view<real_type, DeviceType>& tol_time,
                            const Tines::value_type_2d_view<real_type, DeviceType>& fac,
                            const Tines::value_type_1d_view<time_advance_type, DeviceType>& tadv,
                            const Tines::value_type_2d_view<real_type, DeviceType>& state,
                            /// output
                            const Tines::value_type_1d_view<real_type, DeviceType>& t_out,
                            const Tines::value_type_1d_view<real_type, DeviceType>& dt_out,
                            const Tines::value_type_2d_view<real_type, DeviceType>& state_out,
                            /// const data from kinetic model
                            const Tines::value_type_1d_view<KineticModelConstData<DeviceType>,DeviceType>& kmcds)
    {
        Kokkos::Profiling::pushRegion(profile_name);
        using policy_type = PolicyType;

        using RealType0DViewType = Tines::value_type_0d_view<real_type, DeviceType>;
        using RealType1DViewType = Tines::value_type_1d_view<real_type, DeviceType>;

        auto kmcd_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Kokkos::subview(kmcds, 0));

        const ordinal_type level = 1;
        const ordinal_type per_team_extent = IgnitionZeroDSoot::getWorkSpaceSize(kmcd_host());

        Kokkos::parallel_for
          (profile_name,
           policy,
           KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const ordinal_type i = member.league_rank();
            const auto kmcd_at_i = (kmcds.extent(0) == 1 ? kmcds(0) : kmcds(i));
            const auto tadv_at_i = tadv(i);
            const real_type t_end = tadv_at_i._tend;
            const RealType0DViewType t_out_at_i = Kokkos::subview(t_out, i);
            if (t_out_at_i() < t_end) {
              const RealType1DViewType state_at_i = Kokkos::subview(state, i, Kokkos::ALL());
              const RealType1DViewType state_out_at_i_Gas = Kokkos::subview(state_out,i,std::make_pair(0,3+kmcd_at_i.nSpec));
              const RealType1DViewType state_out_at_i = Kokkos::subview(state_out, i, Kokkos::ALL());
              const RealType1DViewType fac_at_i = Kokkos::subview(fac, i, Kokkos::ALL());

              const RealType0DViewType dt_out_at_i = Kokkos::subview(dt_out, i);
              Scratch<RealType1DViewType> work(member.team_scratch(level),
                                               per_team_extent);
              //can create subview here or just keep going
//              Impl::StateVector<RealType1DViewType> sv_at_i(kmcd_at_i.nSpec, state_at_i);
              Impl::StateVector<RealType1DViewType> sv_out_at_i(kmcd_at_i.nSpec, state_out_at_i_Gas);
//              TCHEM_CHECK_ERROR(!sv_at_i.isValid(), "Error: input state vector is not valid");
              TCHEM_CHECK_ERROR(!sv_out_at_i.isValid(), "Error: input state vector is not valid");
              {
                const ordinal_type jacobian_interval = tadv_at_i._jacobian_interval;
                const ordinal_type max_num_newton_iterations = tadv_at_i._max_num_newton_iterations;
                const ordinal_type max_num_time_iterations = tadv_at_i._num_time_iterations_per_interval;

                const real_type
                  dt_in = tadv_at_i._dt,
                  dt_min = tadv_at_i._dtmin,
                  dt_max = tadv_at_i._dtmax;
                const real_type t_beg = tadv_at_i._tbeg;

                const auto temperature = state_at_i(2);
                const auto pressure = state_at_i(1);
//                const auto Ys = sv_at_i.MassFractions();
//                using range_type = Kokkos::pair<ordinal_type, ordinal_type>;
//                const auto Ys = Kokkos::subview(state_at_i, 0, range_type(3, 3 + kmcd_at_i.nSpec));
                const auto YCarbon = state_at_i(3+kmcd_at_i.nSpec);
                const auto SootNumberDensity = state_at_i(4+kmcd_at_i.nSpec);

                const RealType0DViewType temperature_out(sv_out_at_i.TemperaturePtr());
                const RealType0DViewType pressure_out(sv_out_at_i.PressurePtr());
                const RealType0DViewType density_out(sv_out_at_i.DensityPtr());
                const RealType0DViewType YCarbon_out(&state_out_at_i(3+kmcd_at_i.nSpec));
                const RealType0DViewType SootNumberDensity_out(&state_out_at_i(4+kmcd_at_i.nSpec));
                const RealType1DViewType Ys_out= sv_out_at_i.MassFractions();

                const ordinal_type m = kmcd_at_i.nSpec + 3; //Changed this, this should be number of vars in source term?
                auto wptr = work.data();
                const RealType1DViewType vals(wptr, m);
                wptr += m;
                const RealType1DViewType ww(wptr,
                                            work.extent(0) - (wptr - work.data()));

                /// we can only guarantee vals is contiguous array. we basically assume
                /// that a state vector can be arbitrary ordered.

                /// m is nSpec + 3 // temp Yi carbon Nd
                Kokkos::parallel_for
                  (Kokkos::TeamVectorRange(member, m),
                   [&](const ordinal_type& i) {
                     vals(i) = (i == 0 ? temperature :
                                (i < kmcd_at_i.nSpec +1 ? state_at_i(i+2) :
                                (i == kmcd_at_i.nSpec+1 ? YCarbon : SootNumberDensity) ) );
                   });
                member.team_barrier();

                IgnitionZeroDSootImplementation
                  ::team_invoke(member,
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
                                pressure,
                                vals,
                                t_out_at_i,
                                dt_out_at_i,
                                pressure_out,
                                vals,
                                ww,
                                kmcd_at_i);

                member.team_barrier();
                //save the new values to the out variables
                Kokkos::parallel_for
                  (Kokkos::TeamVectorRange(member, m),
                   [&](const ordinal_type& i) {
                     if (i == 0)
                       temperature_out() = vals(0);
                     else if (i < kmcd_at_i.nSpec +1)
                       Ys_out(i - 1) = vals(i);
                     else if (i == kmcd_at_i.nSpec + 1)
                         YCarbon_out() = vals(i);
                     else
                         SootNumberDensity_out() = vals(i);
                   });

                member.team_barrier();
                //Calculate the new density based on new state
                //TODO:: This might have to be changed, have to think about how we're solving this
                density_out() = Impl::RhoMixMs<real_type,DeviceType>
                  ::team_invoke(member, temperature_out(),
                                pressure_out(), Ys_out, kmcd_at_i);
                member.team_barrier();
              }
            }
          });
        Kokkos::Profiling::popRegion();
    }

  
} // namespace TChem
