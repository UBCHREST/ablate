#ifndef ABLATELIBRARY_MOCKFACTORY_HPP
#define ABLATELIBRARY_MOCKFACTORY_HPP

#include <map>
#include <string>
#include "gmock/gmock.h"
#include "parser/factory.hpp"

using namespace ablate::parser;

namespace ablateTesting::parser {

MATCHER_P(NameIs, name, "") { return (arg.inputName == name); }

class MockFactory : public ablate::parser::Factory {
   public:
    MOCK_METHOD(std::shared_ptr<Factory>, GetFactory, (const std::string& name), (override, const));
    MOCK_METHOD(std::vector<std::shared_ptr<ablate::parser::Factory>>, GetFactorySequence, (const std::string& name), (override, const));
    MOCK_METHOD(const std::string&, GetClassType, (), (override, const));
    MOCK_METHOD(std::string, Get, (const ArgumentIdentifier<std::string>&), (override, const));
    MOCK_METHOD(int, Get, (const ArgumentIdentifier<int>&), (override, const));
    MOCK_METHOD(double, Get, (const ArgumentIdentifier<double>&), (override, const));
    MOCK_METHOD(std::vector<std::string>, Get, (const ArgumentIdentifier<std::vector<std::string>>&), (override, const));
    MOCK_METHOD(std::vector<int>, Get, (const ArgumentIdentifier<std::vector<int>>&), (override, const));
    MOCK_METHOD(std::vector<double>, Get, (const ArgumentIdentifier<std::vector<double>>&), (override, const));
    MOCK_METHOD(std::filesystem::path, Get, (const ArgumentIdentifier<std::filesystem::path>&), (override, const));
    MOCK_METHOD(bool, Get, (const ArgumentIdentifier<bool>&), (override, const));
    MOCK_METHOD((std::map<std::string, std::string>), Get, ((const ArgumentIdentifier<std::map<std::string, std::string>>&)), (override, const));
    MOCK_METHOD(bool, Contains, (const std::string& name), (override, const));
    MOCK_METHOD(std::unordered_set<std::string>, GetKeys, (), (const, override));
};
}  // namespace ablateTesting::parser

#endif  // ABLATELIBRARY_MOCKFACTORY_HPP
