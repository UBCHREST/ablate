#ifndef ABLATELIBRARY_EOS_HPP
#define ABLATELIBRARY_EOS_HPP
#include <petsc.h>
#include <map>
#include "eos.h"

namespace ablate::eos {
class EOS {
   protected:
    EOSData eosData;
    PetscOptions eosOptions;

   public:
    EOS(std::string type, std::map<std::string, std::string> parameters);
    virtual ~EOS();

    inline EOSData GetEOSData() { return eosData; }
};
}  // namespace ablate::eos

#endif  // ABLATELIBRARY_EOS_HPP
