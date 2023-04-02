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
#ifndef ABLATELIBRARY_IGNITION_ZEROD_SOOT_PROBLEM_HPP
#ifndef KOKKOS_ENABLE_CUDA
#define ABLATELIBRARY_IGNITION_ZEROD_SOOT_PROBLEM_HPP

#include "Tines_Internal.hpp"

#include "Soot7StepReactionModel.hpp"
#include "TChem_Impl_JacobianReduced.hpp"
#include "TChem_Impl_SourceTerm.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"
#include "eos/tChemSoot/temperatureFcn.hpp"

namespace ablate::eos::tChemSoot {
template <typename ValueType, typename DeviceType>
struct IgnitionZeroD_ProblemSoot {
    using device_type = DeviceType;
    using real_type_0d_view_type = Tines::value_type_0d_view<real_type, device_type>;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;
    using kinetic_model_type = KineticModelConstData<device_type>;

    real_type_1d_view_type _work;
    real_type_1d_view_type _work_cvode;
    real_type_1d_view_type _fac;
    kinetic_model_type _kmcd;
    real_type _densityTot;  // Total density should be constant
    real_type_1d_view_type _hi_ref;
    real_type_1d_view_type _hi_Scratch;
    real_type_1d_view_type cp_gas_Scratch;

    KOKKOS_DEFAULTED_FUNCTION
    IgnitionZeroD_ProblemSoot() = default;

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getNumberOfTimeODEs(const kinetic_model_type& kmcd) {
        // All species plus Soot and NDD, and Temperature. Temperature actually solved from current internal energy, but need to carry it forward as the source function call still requires the scratch
        // space Since it doesn't know we aren't solving for it
        return kmcd.nSpec + 3;
    }

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getNumberOfConstraints(const kinetic_model_type& kmcd) { return 0; }

    KOKKOS_INLINE_FUNCTION
    static ordinal_type getNumberOfEquations(const kinetic_model_type& kmcd) { return getNumberOfTimeODEs(kmcd) + getNumberOfConstraints(kmcd); }

    KOKKOS_INLINE_FUNCTION
    ordinal_type getNumberOfTimeODEs() const { return getNumberOfTimeODEs(_kmcd); }

    KOKKOS_INLINE_FUNCTION
    ordinal_type getNumberOfConstraints() const { return getNumberOfConstraints(_kmcd); }

    KOKKOS_INLINE_FUNCTION
    ordinal_type getNumberOfEquations() const { return getNumberOfTimeODEs() + getNumberOfConstraints(); }

    KOKKOS_INLINE_FUNCTION
    ordinal_type getWorkSpaceSize() const { return getWorkSpaceSize(_kmcd); }

    KOKKOS_INLINE_FUNCTION  // Probably Need to change this, but I'm not sure rn.... ??? -klb
        static ordinal_type
        getWorkSpaceSize(const kinetic_model_type& kmcd) {
        const ordinal_type src_workspace_size = ::TChem::Impl::SourceTerm<real_type, device_type>::getWorkSpaceSize(kmcd);
        const ordinal_type jac_workspace_size = ::TChem::Impl::JacobianReduced::getWorkSpaceSize(kmcd);
        const ordinal_type workspace_size_analytical_jacobian = (jac_workspace_size > src_workspace_size ? jac_workspace_size : src_workspace_size);

        const ordinal_type m = getNumberOfEquations(kmcd);
        const ordinal_type workspace_size_sacado_numerical_jacobian = src_workspace_size + 2 * m * ats<real_type>::sacadoStorageCapacity();

        const ordinal_type workspace_size =
            (workspace_size_analytical_jacobian > workspace_size_sacado_numerical_jacobian ? workspace_size_analytical_jacobian : workspace_size_sacado_numerical_jacobian);

        return workspace_size;
    }

    KOKKOS_INLINE_FUNCTION
    void setWorkspace(const real_type_1d_view_type& work) { _work = work; }

    /// mxm matrix storage
    KOKKOS_INLINE_FUNCTION
    void setWorkspaceCVODE(const real_type_1d_view_type& work_cvode) { _work_cvode = work_cvode; }

    // MAIN CALL THAT PROBABLY NEEDS TO BE CHANGED, should be able to reuse SourceTerm
    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void computeFunction(const MemberType& member, const real_type_1d_view_type& x, const real_type_1d_view_type& f) const {
        // For soot reaction help, just make sure the carbon and Ndd arent negative, could cause problems in ode stiffness but most likely wont
        x(_kmcd.nSpec + 1) = PetscMax(x(_kmcd.nSpec + 1), 0);  // YCarbon
        x(_kmcd.nSpec + 2) = PetscMax(x(_kmcd.nSpec + 2), 0);  // Ndd

        // Convert total mass fractions to gas mass fractions, and pull out YCarbon and Nd for temperature calculation!
        const real_type_1d_view_type gas_StateVector = real_type_1d_view_type("GaseousSpecies", _kmcd.nSpec + 3);
        real_type Ycarbon = PetscMax(x(_kmcd.nSpec + 1), 0);
        real_type Nd = PetscMax(x(_kmcd.nSpec + 2), 0) * ablate::eos::tChemSoot::NddScaling;

        // Calculate gas density based on total density is constant and back it out from Yc
        gas_StateVector(0) = (1 - Ycarbon) / (1 / _densityTot - Ycarbon / ablate::eos::tChemSoot::solidCarbonDensity);  // gaseous density
        gas_StateVector(2) = x(0);                                                                                      // temperature
        for (int i = 0; i < _kmcd.nSpec; i++) {
            gas_StateVector(i + 3) = x(i + 1) / (1 - Ycarbon);
        }
        const ::TChem::Impl::StateVector gas_SV(_kmcd.nSpec, gas_StateVector);

        // CalculatePressure based on density of the gas and temperature
        gas_SV.Pressure() = ablate::eos::tChem::impl::pressureFcn<real_type, DeviceType>::team_invoke(member, gas_SV, _kmcd);

        // The density is calculated inside below and thus will be the gas density (per unit volume of gas !!0 and thus will return the correct values
        Impl::SourceTerm<real_type, device_type>::team_invoke(member, gas_SV.Temperature(), gas_SV.Pressure(), gas_SV.MassFractions(), f, _work, _kmcd);

        // Now add in the correct source terms due to the soot reaction rates here (Also Adjusts it for SVF!)
        Soot7StepReactionModel::UpdateSourceWithSootMechanismRatesTemperature<device_type>(member, f, Ycarbon, Nd, gas_SV, _densityTot, cp_gas_Scratch, _hi_Scratch, _kmcd);
    }

    /// this one is used in time integration nonlinear solve
    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void computeJacobian(const MemberType& member, const real_type_1d_view_type& s, const real_type_2d_view_type& J) const {
        //            computeAnalyticalJacobian(member, s, J);
        computeNumericalJacobian(member, s, J);
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void computeNumericalJacobian(const MemberType& member, const real_type_1d_view_type& x, const real_type_2d_view_type& J) const {
        const ordinal_type m = getNumberOfEquations();
        /// _work is used for evaluating a function
        /// f_0 and f_h should be gained from the tail
        real_type* wptr = _work.data() + (_work.span() - 2 * m);
        real_type_1d_view_type f_0(wptr, m);
        wptr += f_0.span();
        real_type_1d_view_type f_h(wptr, m);
        wptr += f_h.span();

        /// use the default values
        real_type fac_min(-1), fac_max(-1);

        Tines::NumericalJacobianForwardDifference<real_type, device_type>::invoke(member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
        member.team_barrier();
        // NumericalJacobianCentralDifference::team_invoke_detail(
        //   member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
        // NumericalJacobianRichardsonExtrapolation::team_invoke_detail
        //  (member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
    }

    template <typename MemberType>
    KOKKOS_INLINE_FUNCTION void computeNumericalJacobianRichardsonExtrapolation(const MemberType& member, const real_type_1d_view_type& x, const real_type_2d_view_type& J) const {
        const ordinal_type m = getNumberOfEquations();
        /// _work is used for evaluating a function
        /// f_0 and f_h should be gained from the tail
        real_type* wptr = _work.data() + (_work.span() - 2 * m);
        real_type_1d_view_type f_0(wptr, m);
        wptr += f_0.span();
        real_type_1d_view_type f_h(wptr, m);
        wptr += f_h.span();

        /// use the default values
        real_type fac_min(-1), fac_max(-1);
        Tines::NumericalJacobianRichardsonExtrapolation<real_type, device_type>::invoke(member, *this, fac_min, fac_max, _fac, x, f_0, f_h, J);
        member.team_barrier();
    }

    //        template<typename MemberType>
    //        KOKKOS_INLINE_FUNCTION void computeAnalyticalJacobian(const MemberType& member,
    //                                                              const real_type_1d_view_type& x,
    //                                                              const real_type_2d_view_type& J) const {
    //            const real_type t = x(0);
    //            const real_type_1d_view_type Ys(&x(1), _kmcd.nSpec);
    //           Impl::JacobianReduced::team_invoke(member, t, _p, Ys, J, _work, _kmcd);
    //            member.team_barrier();
    //        }
};
}  // namespace ablate::eos::tChemSoot
#endif
#endif
