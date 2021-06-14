#include <fstream>
#include <memory>
#include <sstream>
#include "gtest/gtest.h"
#include "parser/yamlParser.hpp"

namespace ablateTesting::parser {

using namespace ablate::parser;
namespace fs = std::filesystem;

TEST(YamlParserTests, ShouldCreateFromString) {
    // arrange
    std::string yaml = "---\n item: \"im a string!\"";

    // act
    auto yamlParser = std::make_shared<YamlParser>(yaml);

    // assert
    ASSERT_EQ("im a string!", yamlParser->Get(ArgumentIdentifier<std::string>{"item"}));
    ASSERT_EQ("", yamlParser->GetClassType());
}

TEST(YamlParserTests, ShouldCreateFromFile) {
    // arrange
    fs::path tempPath = fs::temp_directory_path();
    tempPath /= "tempFile.yaml";

    std::ofstream ofs(tempPath);
    ofs << "---" << std::endl;
    ofs << " item: \"im a string!\"";
    ofs.close();

    // act
    auto yamlParser = std::make_shared<YamlParser>(tempPath);

    // assert
    ASSERT_EQ("im a string!", yamlParser->Get(ArgumentIdentifier<std::string>{"item"}));
    ASSERT_EQ("", yamlParser->GetClassType());

    // cleanup
    fs::remove(tempPath);
}

TEST(YamlParserTests, ShouldParseStrings) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item: im_a_string " << std::endl;
    yaml << " item 2: im a string " << std::endl;
    yaml << " item3: \"im a string \" " << std::endl;

    // act
    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // assert
    ASSERT_EQ("im_a_string", yamlParser->Get(ArgumentIdentifier<std::string>{"item"}));
    ASSERT_EQ("im a string", yamlParser->Get(ArgumentIdentifier<std::string>{"item 2"}));
    ASSERT_EQ("im a string ", yamlParser->Get(ArgumentIdentifier<std::string>{"item3"}));
    ASSERT_EQ("", yamlParser->GetClassType());
}

TEST(YamlParserTests, ShouldThrowErrorForMissingString) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item: im_a_string " << std::endl;

    // act
    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // assert
    ASSERT_THROW(yamlParser->Get(ArgumentIdentifier<std::string>{"itemNotThere"}), std::invalid_argument);
}

TEST(YamlParserTests, ShouldReturnCorrectStringForOptionalValue) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item: im_a_string " << std::endl;
    std::string emptyString = {};

    // act
    auto yamlParser = std::make_shared<YamlParser>(yaml.str());
    // assert
    ASSERT_EQ("im_a_string", yamlParser->Get(ArgumentIdentifier<std::string>{"item", .optional = true}));
    ASSERT_EQ(emptyString, yamlParser->Get(ArgumentIdentifier<std::string>{"item 2", .optional = true}));
    ASSERT_EQ("", yamlParser->GetClassType());
}

TEST(YamlParserTests, ShouldParseInts) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item: 22" << std::endl;
    yaml << " item 2: 1 " << std::endl;
    yaml << " item3: \"3 \" " << std::endl;
    yaml << " item4: \"not an int \" " << std::endl;

    // act
    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // assert
    ASSERT_EQ(22, yamlParser->Get(ArgumentIdentifier<int>{"item"}));
    ASSERT_EQ(1, yamlParser->Get(ArgumentIdentifier<int>{"item 2"}));
    ASSERT_EQ(3, yamlParser->Get(ArgumentIdentifier<int>{"item3"}));
    ASSERT_THROW(yamlParser->Get(ArgumentIdentifier<int>{"item4"}), YAML::BadConversion);
}

TEST(YamlParserTests, ShouldThrowErrorForMissingInt) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item: im_a_string " << std::endl;

    // act
    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // assert
    ASSERT_THROW(yamlParser->Get(ArgumentIdentifier<std::string>{"itemNotThere"}), std::invalid_argument);
}

TEST(YamlParserTests, ShouldReturnCorrectInForOptionalValue) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item: 22" << std::endl;
    int defaultValue = {};

    // act
    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // assert
    ASSERT_EQ(22, yamlParser->Get(ArgumentIdentifier<int>{"item", .optional = true}));
    ASSERT_EQ(defaultValue, yamlParser->Get(ArgumentIdentifier<int>{"item 2", .optional = true}));
}

TEST(YamlParserTests, ShouldParseBools) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item: true" << std::endl;
    yaml << " item 2: False " << std::endl;
    yaml << " item3: false " << std::endl;
    yaml << " item4: \"truafeae \" " << std::endl;
    yaml << " item5: True " << std::endl;
    yaml << " item6: true " << std::endl;

    // act
    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // assert
    ASSERT_EQ(true, yamlParser->Get(ArgumentIdentifier<bool>{"item"}));
    ASSERT_EQ(false, yamlParser->Get(ArgumentIdentifier<bool>{"item 2"}));
    ASSERT_EQ(false, yamlParser->Get(ArgumentIdentifier<bool>{"item3"}));
    ASSERT_THROW(yamlParser->Get(ArgumentIdentifier<bool>{"item4"}), YAML::BadConversion);
    ASSERT_EQ(true, yamlParser->Get(ArgumentIdentifier<bool>{"item5"}));
    ASSERT_EQ(true, yamlParser->Get(ArgumentIdentifier<bool>{"item6"}));
}

TEST(YamlParserTests, ShouldThrowErrorForMissingBool) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item: im_a_string " << std::endl;

    // act
    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // assert
    ASSERT_THROW(yamlParser->Get(ArgumentIdentifier<bool>{"itemNotThere"}), std::invalid_argument);
}

TEST(YamlParserTests, ShouldReturnCorrectBoolForOptionalValue) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item: true" << std::endl;
    bool defaultValue = {};

    // act
    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // assert
    ASSERT_EQ(true, yamlParser->Get(ArgumentIdentifier<bool>{"item", .optional = true}));
    ASSERT_EQ(defaultValue, yamlParser->Get(ArgumentIdentifier<int>{"item 2", .optional = true}));
}

TEST(YamlParserTests, ShouldParseSubFactory) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item: 22" << std::endl;
    yaml << " item 2: " << std::endl;
    yaml << "   child1: 12  " << std::endl;
    yaml << "   child2: im a string " << std::endl;

    // act
    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    auto child = yamlParser->GetFactory("item 2");

    // assert
    ASSERT_EQ(12, child->Get(ArgumentIdentifier<int>{"child1"}));
    ASSERT_EQ("im a string", child->Get(ArgumentIdentifier<std::string>{"child2"}));
    ASSERT_EQ("", child->GetClassType());
}

TEST(YamlParserTests, ShouldParseSubFactoryWithTag) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item: 22" << std::endl;
    yaml << " item 2: !ablate::info::green" << std::endl;
    yaml << "   child1: 12  " << std::endl;
    yaml << "   child2: im a string " << std::endl;

    // act
    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    auto child = yamlParser->GetFactory("item 2");

    // assert
    ASSERT_EQ(12, child->Get(ArgumentIdentifier<int>{"child1"}));
    ASSERT_EQ("im a string", child->Get(ArgumentIdentifier<std::string>{"child2"}));
    ASSERT_EQ("ablate::info::green", child->GetClassType());
}

TEST(YamlParserTests, ShouldThrowErrorForMissingChild) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item: 22" << std::endl;
    yaml << " item 2: !ablate::info::green" << std::endl;
    yaml << "   child1: 12  " << std::endl;
    yaml << "   child2: im a string " << std::endl;

    // act
    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // assert
    ASSERT_THROW(yamlParser->GetFactory("item 3"), std::invalid_argument);
}

TEST(YamlParserTests, ShouldOnlyCreateOneChildInstance) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item: 22" << std::endl;
    yaml << " item 2: !ablate::info::green" << std::endl;
    yaml << "   child1: 12  " << std::endl;
    yaml << "   child2: im a string " << std::endl;

    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // act
    auto childA = yamlParser->GetFactory("item 2");
    auto childB = yamlParser->GetFactory("item 2");

    // assert
    ASSERT_EQ(childA, childB);
}

TEST(YamlParserTests, ShouldReportUnusedChildren) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item1: 22" << std::endl;
    yaml << " item2: !ablate::info::green" << std::endl;
    yaml << "   child1: 12  " << std::endl;
    yaml << "   child2: im a string " << std::endl;
    yaml << " item3:" << std::endl;
    yaml << "   child1: im a string " << std::endl;
    yaml << "   child2: im a string " << std::endl;
    yaml << " item4: 24" << std::endl;
    yaml << " item5:" << std::endl;
    yaml << "   child1: " << std::endl;
    yaml << "     childchild1: 1 " << std::endl;
    yaml << "     childchild2: 1 " << std::endl;
    yaml << "   child2: im a string " << std::endl;
    yaml << " item6:" << std::endl;
    yaml << "   -           " << std::endl;
    yaml << "      child1: 1 " << std::endl;
    yaml << "      child2: 2" << std::endl;
    yaml << "   -           " << std::endl;
    yaml << "      child1: 3" << std::endl;
    yaml << "      child2: 4" << std::endl;

    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // act
    yamlParser->Get(ArgumentIdentifier<int>{"item1"});
    yamlParser->GetFactory("item2")->Get(ArgumentIdentifier<std::string>{"child1"});
    yamlParser->GetFactory("item5")->GetFactory("child1")->Get(ArgumentIdentifier<int>{"childchild1"});
    yamlParser->GetFactorySequence("item6")[0]->Get(ArgumentIdentifier<int>{"child2"});

    // assert
    auto unusedValues = yamlParser->GetUnusedValues();
    ASSERT_EQ(8, unusedValues.size());
    ASSERT_EQ("root/item3", unusedValues[0]);
    ASSERT_EQ("root/item4", unusedValues[1]);
    ASSERT_EQ("root/item2/child2", unusedValues[2]);
    ASSERT_EQ("root/item5/child2", unusedValues[3]);
    ASSERT_EQ("root/item5/child1/childchild2", unusedValues[4]);
    ASSERT_EQ("root/item6/0/child1", unusedValues[5]);
    ASSERT_EQ("root/item6/1/child1", unusedValues[6]);
    ASSERT_EQ("root/item6/1/child2", unusedValues[7]);
}

TEST(YamlParserTests, ShouldGetListOfStrings) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item1: 22" << std::endl;
    yaml << " item2:" << std::endl;
    yaml << "   - string 1  " << std::endl;
    yaml << "   - string 2 " << std::endl;
    yaml << "   - string 3 " << std::endl;

    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // act
    auto list = yamlParser->Get(ArgumentIdentifier<std::vector<std::string>>{"item2"});

    // assert
    std::vector<std::string> expectedValues = {"string 1", "string 2", "string 3"};
    ASSERT_EQ(list, expectedValues);
}

TEST(YamlParserTests, ShouldGetCorrectValueForOptionalListOfStrings) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item1: 22" << std::endl;

    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // act
    auto list = yamlParser->Get(ArgumentIdentifier<std::vector<std::string>>{"item2", .optional = true});

    // assert
    std::vector<std::string> expectedValues = {};
    ASSERT_EQ(list, expectedValues);
}

TEST(YamlParserTests, ShouldGetListOfInts) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item1: 22" << std::endl;
    yaml << " item2:" << std::endl;
    yaml << "   - 1  " << std::endl;
    yaml << "   - 2 " << std::endl;
    yaml << "   - 3 " << std::endl;
    yaml << " item3: [4, 5, 6]" << std::endl;

    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // act
    auto list1 = yamlParser->Get(ArgumentIdentifier<std::vector<int>>{"item2"});
    auto list2 = yamlParser->Get(ArgumentIdentifier<std::vector<int>>{"item3"});

    // assert
    std::vector<int> expectedValues1 = {1, 2, 3};
    ASSERT_EQ(list1, expectedValues1);
    std::vector<int> expectedValues2 = {4, 5, 6};
    ASSERT_EQ(list2, expectedValues2);
}

TEST(YamlParserTests, ShouldGetCorrectValueForOptionalListOfInt) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item1: 22" << std::endl;

    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // act
    auto list = yamlParser->Get(ArgumentIdentifier<std::vector<int>>{"item2", .optional = true});

    // assert
    std::vector<int> expectedValues = {};
    ASSERT_EQ(list, expectedValues);
}

TEST(YamlParserTests, ShouldGetListOfDouble) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item1: 22" << std::endl;
    yaml << " item2:" << std::endl;
    yaml << "   - 1.1  " << std::endl;
    yaml << "   - 2 " << std::endl;
    yaml << "   - 3.3 " << std::endl;
    yaml << " item3: [4.4, 5, 6.6]" << std::endl;

    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // act
    auto list1 = yamlParser->Get(ArgumentIdentifier<std::vector<double>>{"item2"});
    auto list2 = yamlParser->Get(ArgumentIdentifier<std::vector<double>>{"item3"});

    // assert
    std::vector<double> expectedValues1 = {1.1, 2, 3.3};
    ASSERT_EQ(list1, expectedValues1);
    std::vector<double> expectedValues2 = {4.4, 5, 6.6};
    ASSERT_EQ(list2, expectedValues2);
}

TEST(YamlParserTests, ShouldGetCorrectValueForOptionalListOfDouble) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item1: 22" << std::endl;

    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // act
    auto list = yamlParser->Get(ArgumentIdentifier<std::vector<int>>{"item2", .optional = true});

    // assert
    std::vector<int> expectedValues = {};
    ASSERT_EQ(list, expectedValues);
}

TEST(YamlParserTests, ShouldGetMapOfStrings) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item1: 22" << std::endl;
    yaml << " item2:" << std::endl;
    yaml << "   string1: 1  " << std::endl;
    yaml << "   string2: 2 " << std::endl;
    yaml << "   string3: 3 " << std::endl;

    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // act
    auto map = yamlParser->Get(ArgumentIdentifier<std::map<std::string, std::string>>{"item2"});

    // assert
    std::map<std::string, std::string> expectedValues = {{"string1", "1"}, {"string2", "2"}, {"string3", "3"}};
    ASSERT_EQ(map, expectedValues);
}

TEST(YamlParserTests, ShouldGetCorrectValueForOptionalMap) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item1: 22" << std::endl;

    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // act
    auto map = yamlParser->Get(ArgumentIdentifier<std::map<std::string, std::string>>{"item2", .optional = true});

    // assert
    std::map<std::string, std::string> expectedValues = {};
    ASSERT_EQ(map, expectedValues);
}

TEST(YamlParserTests, ShouldGetListOfFactories) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item1: 22" << std::endl;
    yaml << " item2:" << std::endl;
    yaml << "   -           " << std::endl;
    yaml << "      child1: 1 " << std::endl;
    yaml << "      child2: 2" << std::endl;
    yaml << "   - !ablate::info::green " << std::endl;
    yaml << "      child1: 3" << std::endl;
    yaml << "      child2: 4" << std::endl;

    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // act
    auto list = yamlParser->GetFactorySequence("item2");

    // assert
    ASSERT_EQ(list[0]->Get(ArgumentIdentifier<int>{"child1"}), 1);
    ASSERT_EQ(list[0]->Get(ArgumentIdentifier<int>{"child2"}), 2);
    ASSERT_EQ(list[0]->GetClassType(), "");
    ASSERT_EQ(list[1]->Get(ArgumentIdentifier<int>{"child1"}), 3);
    ASSERT_EQ(list[1]->Get(ArgumentIdentifier<int>{"child2"}), 4);
    ASSERT_EQ(list[1]->GetClassType(), "ablate::info::green");
}

TEST(YamlParserTests, ShouldGetListOfKeys) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item1: 22" << std::endl;
    yaml << " item2:" << std::endl;
    yaml << "      child1: 1 " << std::endl;
    yaml << "      child2: 2" << std::endl;

    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // act
    auto keysRoot = yamlParser->GetKeys();
    auto childKeys = yamlParser->GetFactory("item2")->GetKeys();

    // assert
    ASSERT_EQ(keysRoot.size(), 2);
    ASSERT_EQ(keysRoot.count("item1"), 1);
    ASSERT_EQ(keysRoot.count("item2"), 1);

    ASSERT_EQ(childKeys.size(), 2);
    ASSERT_EQ(childKeys.count("child1"), 1);
    ASSERT_EQ(childKeys.count("child2"), 1);
}

TEST(YamlParserTests, ShouldGetListAsString) {
    // arrange
    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << " item1: 22" << std::endl;
    yaml << " item2:" << std::endl;
    yaml << "   - 1.1  " << std::endl;
    yaml << "   - 2 " << std::endl;
    yaml << "   - 3.3 " << std::endl;
    yaml << " item3: [4.4, 5, 6.6]" << std::endl;

    auto yamlParser = std::make_shared<YamlParser>(yaml.str());

    // act
    auto list1 = yamlParser->Get(ArgumentIdentifier<std::string>{"item2"});
    auto list2 = yamlParser->Get(ArgumentIdentifier<std::string>{"item3"});

    // assert
    std::string expectedValues1 = "1.1 2 3.3 ";
    ASSERT_EQ(list1, expectedValues1);
    std::string expectedValues2 = "4.4 5 6.6 ";
    ASSERT_EQ(list2, expectedValues2);
}

}  // namespace ablateTesting::parser