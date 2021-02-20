#ifndef ABLATELIBRARY_MOCKFACTORY_HPP
#define ABLATELIBRARY_MOCKFACTORY_HPP

#include "parser/factory.hpp"
#include <map>
#include <string>

namespace ablateTesting::parser {
using namespace ablate::parser;

class MockFactory : public ablate::parser::Factory {
   public:
    MOCK_METHOD(std::shared_ptr<Factory>, GetFactory, (const std::string& name), (override, const));
    MOCK_METHOD(std::vector<std::shared_ptr<ablate::parser::Factory>>, GetFactorySequence, (const std::string& name), (override, const));
    MOCK_METHOD(const std::string&, GetClassType, (), (override, const));
    MOCK_METHOD(std::string, Get, (const ArgumentIdentifier<std::string>&), (override, const));
    MOCK_METHOD(int, Get, (const ArgumentIdentifier<int>&), (override, const));
    MOCK_METHOD(std::vector<std::string>, Get, (const ArgumentIdentifier<std::vector<std::string>>&), (override, const));
    MOCK_METHOD((std::map<std::string,std::string>), Get, ((const ArgumentIdentifier<std::map<std::string,std::string>>&)), (override, const));
};
}

#endif  // ABLATELIBRARY_MOCKFACTORY_HPP
