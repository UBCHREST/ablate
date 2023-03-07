#ifndef ABLATELIBRARY_STATEVECTORSOOT_HPP
#define ABLATELIBRARY_STATEVECTORSOOT_HPP
#include "TChem_Util.hpp"
#include "sootConstants.hpp"

namespace ablate::eos::tChemSoot {

/// state vector standard interface defining the internal structure
/// 0. Density (R)
/// 1. Pressure (P)
/// 2. Temperature (T)
/// 3. MassFractions (Yn)
template <typename RealType1DView>
struct StateVectorSoot {
   private:
    const ::TChem::ordinal_type _nSpec;
    const RealType1DView _v;

   public:
    using real_value_type = typename RealType1DView::non_const_value_type;
    using range_type = Kokkos::pair<::TChem::ordinal_type, ::TChem::ordinal_type>;

    KOKKOS_INLINE_FUNCTION StateVectorSoot(const ::TChem::ordinal_type nGasSpec, const RealType1DView &v) : _nSpec(nGasSpec), _v(v) {}

    /// validate input vector
    KOKKOS_INLINE_FUNCTION bool isValid() const {
        const bool is_valid_rank = (RealType1DView::Rank == 1);
        const bool is_extent_valid = (_v.extent(0) <= (3 + _nSpec + 1 /** for solid carbon **/));
        return (is_valid_rank && is_extent_valid);
    }

    /// assign a pointer to update the vector in a batch fashion
    KOKKOS_INLINE_FUNCTION ::TChem::ordinal_type size() const { return _nSpec + 3 + 1; }
    KOKKOS_INLINE_FUNCTION void assign_data(real_value_type *ptr) { _v.assign_data(ptr); }

    /// copy access to private members
    KOKKOS_INLINE_FUNCTION RealType1DView KokkosView() const { return _v; }
    KOKKOS_INLINE_FUNCTION ::TChem::ordinal_type NumGasSpecies() const { return _nSpec; }
    KOKKOS_INLINE_FUNCTION ::TChem::ordinal_type NumSpecies() const { return _nSpec + 1; }

    /// interface to state vector
    KOKKOS_INLINE_FUNCTION real_value_type &Density() const { return _v(0); }
    KOKKOS_INLINE_FUNCTION real_value_type &Pressure() const { return _v(1); }
    KOKKOS_INLINE_FUNCTION real_value_type &Temperature() const { return _v(2); }
    KOKKOS_INLINE_FUNCTION auto MassFractions() const -> decltype(Kokkos::subview(_v, range_type(3, 3 + _nSpec + 1))) { return Kokkos::subview(_v, range_type(3, 3 + _nSpec + 1)); }
    KOKKOS_INLINE_FUNCTION real_value_type &MassFractionCarbon() const { return _v(3 + _nSpec); }
    KOKKOS_INLINE_FUNCTION real_value_type &SootNumberDensity() const { return _v(3 + _nSpec + 1); }

    KOKKOS_INLINE_FUNCTION real_value_type *DensityPtr() const { return &_v(0); }
    KOKKOS_INLINE_FUNCTION real_value_type *PressurePtr() const { return &_v(1); }
    KOKKOS_INLINE_FUNCTION real_value_type *TemperaturePtr() const { return &_v(2); }
    KOKKOS_INLINE_FUNCTION real_value_type *MassFractionsPtr() const { return &_v(3); }

    // Helper Function to split the total state vector into an appropriate gaseous state vector
    // Currently assumes all species were already normalized
    template <typename real_1d_viewType>
    inline void SplitYiState(Impl::StateVector<real_1d_viewType> &gaseousState) const {
        double Yc = MassFractionCarbon();

        // pressure, temperature (assumed the same in both phases)
        gaseousState.Temperature() = Temperature();
        gaseousState.Pressure() = Pressure();
        auto yiGaseousState = gaseousState.MassFractions();
        auto yi = MassFractions();

        for (auto ns = 0; ns < _nSpec; ns++) {
            yiGaseousState(ns) = yi(ns) / (1.0 - Yc);
        }
        // Need to calculate the gaseous density at this state
        gaseousState.Density() = (1 - Yc) / (1 / Density() - Yc / ablate::eos::tChemSoot::solidCarbonDensity);
    }
};

static KOKKOS_INLINE_FUNCTION ordinal_type getStateVectorSootSize(const ordinal_type nGasSpec) { return nGasSpec + 3 + 2 /*yc, ndd */; }

}  // namespace ablate::eos::tChemSoot
#endif  // ABLATELIBRARY_STATEVECTORSOOT_HPP
