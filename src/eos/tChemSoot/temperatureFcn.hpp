#ifndef ABLATE_SOOTTEMPERATUREFCN_HPP
#define ABLATE_SOOTTEMPERATUREFCN_HPP
#include <Kokkos_Macros.hpp>
#ifndef KOKKOS_ENABLE_CUDA

#include <TChem_Impl_RhoMixMs.hpp>
#include "TChem_KineticModelData.hpp"
#include "eos/tChemSoot.hpp"
#include "eos/tChemSoot/sensibleInternalEnergyFcn.hpp"
namespace ablate::eos::tChemSoot::impl {

template <typename ValueType, typename DeviceType>
struct TemperatureFcn {
    using value_type = ValueType;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, DeviceType>;

    /**
     * tchem like function that computes the temperature using an iterative method assuming the interna energy is known.
     * @tparam MemberType
     * @tparam KineticModelConstDataType
     * @param member
     * @param svGas
     * @param YCarbon
     * @param kmcd
     * @return
     */
    template <typename MemberType, typename KineticModelConstDataType>
    KOKKOS_INLINE_FUNCTION static value_type team_invoke(const MemberType& member,
                                                         /// input
                                                         const Impl::StateVector<Tines::value_type_1d_view<real_type, DeviceType>> svGas,  // Gaseous State Vector
                                                         const real_type Yc, const Tines::value_type_0d_view<real_type, DeviceType> internalEnergyRef, const real_type_1d_view_type hi_Scratch,
                                                         const real_type_1d_view_type cp_gas_Scratch, const real_type_1d_view_type hi_Ref_Values, const KineticModelConstDataType& kmcd) {
        TCHEM_CHECK_ERROR(!svGas.isValid(), "Error: input state vector is not valid");
        {
            real_type& t = svGas.Temperature();
            double t2 = t;
            const real_type_1d_view_type ys_gas = svGas.MassFractions();

            // set some constants
            const auto EPS_T_RHO_E = 1;
            const auto ITERMAX_T = 100;

            // compute the first error
            double e2 = ablate::eos::tChemSoot::impl::SensibleInternalEnergyFcn<real_type, DeviceType>::team_invoke(member, Yc, svGas, hi_Scratch, cp_gas_Scratch, hi_Ref_Values, kmcd);
            double f2 = internalEnergyRef() - e2;
            if (Tines::ats<real_type>::abs(f2) > EPS_T_RHO_E) {
                double t0 = t2;
                double f0 = f2;
                double t1 = t0 + 1;

                t = t1;
                double e1 = ablate::eos::tChemSoot::impl::SensibleInternalEnergyFcn<real_type, DeviceType>::team_invoke(member, Yc, svGas, hi_Scratch, cp_gas_Scratch, hi_Ref_Values, kmcd);
                double f1 = internalEnergyRef() - e1;

                for (int it = 0; it < ITERMAX_T; it++) {
                    t2 = t1 - f1 * (t1 - t0) / (f1 - f0 + 1E-30);
                    t2 = std::max(1.0, t2);
                    t = t2;
                    e2 = ablate::eos::tChemSoot::impl::SensibleInternalEnergyFcn<real_type, DeviceType>::team_invoke(member, Yc, svGas, hi_Scratch, cp_gas_Scratch, hi_Ref_Values, kmcd);
                    f2 = internalEnergyRef() - e2;
                    if (Tines::ats<real_type>::abs(f2) <= EPS_T_RHO_E) {
                        t = t2;
                        // Make Sure Temperature is set in the stateSpace
                        return svGas.Temperature();
                    }
                    t0 = t1;
                    t1 = t2;
                    f0 = f1;
                    f1 = f2;
                }
                t = t2;
            }
        }
        // Make Sure Temperature is set in the stateSpace
        return svGas.Temperature();
    }
};

}  // namespace ablate::eos::tChemSoot::impl
#endif
#endif
