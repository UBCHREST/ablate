#include <memory>
#include "PetscTestFixture.hpp"
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "mathFunctions/geom/box.hpp"
#include "mathFunctions/geom/cylinder.hpp"
#include "mathFunctions/geom/cylinderShell.hpp"
#include "mathFunctions/geom/difference.hpp"
#include "mathFunctions/geom/geometry.hpp"
#include "mathFunctions/geom/inverse.hpp"
#include "mathFunctions/geom/sphere.hpp"
#include "mathFunctions/geom/surface.hpp"
#include "mathFunctions/geom/triangle.hpp"
#include "mathFunctions/geom/union.hpp"

using namespace ablate::mathFunctions::geom;
namespace ablateTesting::mathFunctions::geom {

template <class Value>
struct ExpectedValue {
    std::vector<PetscReal> xyz;
    Value value;
};

struct GeometryTestScalarParameters {
    std::function<std::shared_ptr<Geometry>()> createGeom;
    std::vector<ExpectedValue<double>> expectedResults;
};

class GeometryTestScalarFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<GeometryTestScalarParameters> {};

TEST_P(GeometryTestScalarFixture, ShouldComputeCorrectAnswerFromXYZ) {
    // arrange
    const auto& param = GetParam();
    auto function = param.createGeom();

    for (const auto& [xyz, expectedValue] : param.expectedResults) {
        double x = xyz[0];
        double y = xyz.size() > 1 ? xyz[1] : NAN;
        double z = xyz.size() > 2 ? xyz[2] : NAN;

        // act/assert
        ASSERT_DOUBLE_EQ(expectedValue, function->Eval(x, y, z, NAN)) << " at " << x << ", " << y << ", " << z;
    }
}

TEST_P(GeometryTestScalarFixture, ShouldComputeCorrectAnswerFromCoord) {
    // arrange
    const auto& param = GetParam();
    auto function = param.createGeom();

    // act/assert
    for (const auto& [xyz, expectedValue] : param.expectedResults) {
        ASSERT_DOUBLE_EQ(expectedValue, function->Eval(xyz.data(), xyz.size(), NAN));
    }
}

INSTANTIATE_TEST_SUITE_P(
    GeometryTests, GeometryTestScalarFixture,
    testing::Values(
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{.2, .2}, .1, ablate::mathFunctions::Create(10.0));
                                           },
                                       .expectedResults = {{.xyz = {.25, .25}, .value = 10}}},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{.2, .2}, .1, ablate::mathFunctions::Create(10.0));
                                           },
                                       .expectedResults = {{.xyz = {.3, .3}, .value = 0}}},
        (GeometryTestScalarParameters){.createGeom = []() { return std::make_shared<Sphere>(std::vector<double>{0.0}, .1, ablate::mathFunctions::Create(10.0)); },
                                       .expectedResults = {{.xyz = {.1}, .value = 10}}},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{-1, -2, -3}, 2, ablate::mathFunctions::Create(10.0), ablate::mathFunctions::Create(4.2));
                                           },
                                       .expectedResults = {{.xyz = {10, 11, 12}, .value = 4.2}}},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{0.0, 0.0, 1.99}, 2, ablate::mathFunctions::Create(20.0), ablate::mathFunctions::Create(4.2));
                                           },
                                       .expectedResults = {{.xyz = {0.0, 0.0, 2.0}, .value = 20}}},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{.2, .1}, std::vector<double>{.3, .2}, ablate::mathFunctions::Create(10.0));
                                           },
                                       .expectedResults = {{.xyz = {.25, .15}, .value = 10}}},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{.2, .1}, std::vector<double>{.3, .2}, ablate::mathFunctions::Create(10.0));
                                           },
                                       .expectedResults = {{.xyz = {.15, .15}, .value = 0}}},
        (GeometryTestScalarParameters){
            .createGeom =
                []() {
                    return std::make_shared<Box>(std::vector<double>{.2, .1}, std::vector<double>{.3, .2}, ablate::mathFunctions::Create(10.0), ablate::mathFunctions::Create(4.2));
                },
            .expectedResults = {{.xyz = {.25, .25}, .value = 4.2}}},
        (GeometryTestScalarParameters){
            .createGeom =
                []() {
                    return std::make_shared<Box>(std::vector<double>{-2, -2}, std::vector<double>{-1, -1}, ablate::mathFunctions::Create(10.0), ablate::mathFunctions::Create(4.2));
                },
            .expectedResults = {{.xyz = {-1.5, -1.0000001}, .value = 10}}},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{-2, -2}, std::vector<double>{-1, -1});
                                           },
                                       .expectedResults = {{.xyz = {-1.5, -1.0000001, 2}, .value = 1}}},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{-2, -2}, std::vector<double>{-1, -1});
                                           },
                                       .expectedResults = {{.xyz = {-1.5, -2.0000001, 2}, .value = 0}}},
        (GeometryTestScalarParameters){.createGeom = []() { return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step"); },
                                       .expectedResults = {{.xyz = {0.0, 0.0, 0.0}, .value = 1}}},
        (GeometryTestScalarParameters){.createGeom = []() { return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step"); },
                                       .expectedResults = {{.xyz = {0.0, 0.041, 0.0}, /**41 mm should be outside ***/
                                                            .value = 0}}},
        (GeometryTestScalarParameters){
            .createGeom = []() { return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step", ablate::mathFunctions::Create(10.0), ablate::mathFunctions::Create(4.2)); },
            .expectedResults = {{.xyz = {0.005, 0.035, 0.005}, .value = 10}}},
        (GeometryTestScalarParameters){
            .createGeom = []() { return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step", ablate::mathFunctions::Create(10.0), ablate::mathFunctions::Create(4.2)); },
            .expectedResults = {{.xyz = {0.01, 0.035, 0.01}, .value = 4.2}}},
        (GeometryTestScalarParameters){
            .createGeom =
                []() {
                    return std::make_shared<CylinderShell>(std::vector<double>{-0.001, -0.25, 0.1519}, std::vector<double>{-0.001, -0.23, 0.1519}, .1, .2);
                },
            .expectedResults =
                {{.xyz = {0.09353390997023181, -0.237586838224223, 0.10392668643347691}, .value = 1},     {.xyz = {0.16784382726924696, -0.24078984395323688, 0.16180769506641768}, .value = 1},
                 {.xyz = {0.06795544842480095, -0.24119896516384987, 0.2405161434782157}, .value = 1},    {.xyz = {-0.10240145684754887, -0.2347519315508647, 0.19731202506093004}, .value = 1},
                 {.xyz = {-0.13065410514914433, -0.2466488199479725, 0.3032415139585555}, .value = 1},    {.xyz = {0.11338404742910957, -0.24287313833346336, 0.11304931383706829}, .value = 1},
                 {.xyz = {-0.07294125536895812, -0.23408284773497967, 0.005703529030037902}, .value = 1}, {.xyz = {0.11328869080553239, -0.2425755986512529, 0.28721084324462876}, .value = 1},
                 {.xyz = {0.1049742940853976, -0.2410616559964388, 0.13651048929841914}, .value = 1},     {.xyz = {0.0166443880945093, -0.2492774851386651, 0.27864843973915954}, .value = 1},
                 {.xyz = {0.05694208394754052, -0.23842202869792728, 0.25339822636736375}, .value = 1},   {.xyz = {-0.037291395450233966, -0.24857316186017686, 0.03008670922077772}, .value = 1},
                 {.xyz = {0.1771439161925853, -0.2486900482437031, 0.15259685599599748}, .value = 1},     {.xyz = {0.10738635835047228, -0.24139873210027296, 0.04586928741299445}, .value = 1},
                 {.xyz = {0.07156403055912952, -0.15136173333868808, -0.34593954910019187}, .value = 0},  {.xyz = {-0.17995889737037196, -0.3357635106149338, 0.13472779934552137}, .value = 0},
                 {.xyz = {0.39647886810208643, -0.15777978229062772, 0.29838655984614193}, .value = 0},   {.xyz = {0.2065338084143079, -0.23247509509848452, 0.17085116726778693}, .value = 0},
                 {.xyz = {-0.3939149905035768, 0.4840345739239337, 0.3702902271464009}, .value = 0},      {.xyz = {0.17141584334559434, -0.1878204350497401, -0.4657417783545752}, .value = 0},
                 {.xyz = {0.3266427542923126, -0.011672888093754663, 0.21694519404266266}, .value = 0},   {.xyz = {-0.4781415969674002, -0.2512462056016449, 0.35222851934245525}, .value = 0},
                 {.xyz = {0.3377701393009437, -0.010215826152739194, -0.08796550800916347}, .value = 0},  {.xyz = {-0.3568148493635891, -5.2852579357720586E-5, 0.005475293073376131}, .value = 0}}},
        (GeometryTestScalarParameters){
            .createGeom =
                []() {
                    return std::make_shared<CylinderShell>(
                        std::vector<double>{1, 1, 1}, std::vector<double>{0.0, 0.0, 0.0}, .1, .2, ablate::mathFunctions::Create(10.0), ablate::mathFunctions::Create(4.2));
                },
            .expectedResults =
                {{.xyz = {0.5737255071346832, 0.767022803088282, 0.5894503202916701}, .value = 10.0},       {.xyz = {0.9350312671904069, 0.8860091649029023, 1.0366283632890532}, .value = 10.0},
                 {.xyz = {0.028457437231878435, -0.10758872029407462, 0.10596322836141181}, .value = 10.0}, {.xyz = {0.6882172726971765, 0.5592666398031048, 0.7046186970763693}, .value = 10.0},
                 {.xyz = {0.5666051698326102, 0.7263407208655104, 0.7367112249878809}, .value = 10.0},      {.xyz = {0.511871914214413, 0.4666553472974948, 0.7018657959693819}, .value = 10.0},
                 {.xyz = {0.3804938917388385, 0.41052942969170236, 0.23002427367601164}, .value = 10.0},    {.xyz = {0.8033101569961627, 0.9821200990912846, 0.9141486979760338}, .value = 10.0},
                 {.xyz = {0.5503625040357609, 0.7242286828645175, 0.6078857913255158}, .value = 10.0},      {.xyz = {0.4680779770846777, 0.27561050343341265, 0.5031436582795212}, .value = 10.0},
                 {.xyz = {0.03784584080408915, 0.7347327767365763, 0.08749700095536728}, .value = 4.2},     {.xyz = {0.8988149395062159, -0.03626073415056208, 0.8340999023803175}, .value = 4.2},
                 {.xyz = {1.4866610476053264, 0.8405522155957352, -0.32938707732243633}, .value = 4.2},     {.xyz = {-0.3579880975123806, 1.1685083809388985, 0.7102663693480407}, .value = 4.2},
                 {.xyz = {1.4042265778723018, -0.43335570937281354, 1.4878122769063051}, .value = 4.2},     {.xyz = {0.4273613902405924, -0.08227590932321416, -0.4809993582520806}, .value = 4.2},
                 {.xyz = {0.8785875082590098, -0.39824072507040564, 0.9249726070610769}, .value = 4.2},     {.xyz = {0.4367722443348041, 1.316090072841381, 1.1375032842530228}, .value = 4.2},
                 {.xyz = {1.0786837986186593, 0.35271030319116625, -0.09724984881334531}, .value = 4.2},    {.xyz = {1.1199175152976288, -0.3747659414706568, -0.13500782662299615}, .value = 4.2},
                 {.xyz = {-0.2300050335134063, -0.452093404799514, 0.35880262351180936}, .value = 4.2}}},
        (GeometryTestScalarParameters){
            .createGeom =
                []() {
                    return std::make_shared<Cylinder>(std::vector<double>{1, 1, 1}, std::vector<double>{0.0, 0.0, 0.0}, .2, ablate::mathFunctions::Create(10.0), ablate::mathFunctions::Create(4.2));
                },
            .expectedResults = {{.xyz = {.5, .5, .5}, .value = 10.0},
                                {.xyz = {0.3469498160486115, 0.12212685458885075, 0.5935674027949154}, .value = 4.2},
                                {.xyz = {1.422497529062558, 0.0747921342509037, 0.5438022732511887}, .value = 4.2},
                                {.xyz = {-0.3974364402389252, 7.580389995769377E-4, 1.0031405621614435}, .value = 4.2},
                                {.xyz = {0.6201884301886198, -0.38980652133015936, 1.361877392308243}, .value = 4.2},
                                {.xyz = {0.5387681888518165, -0.14883385878364908, -0.05712957242873018}, .value = 4.2},
                                {.xyz = {-0.053761378342362276, 0.905409062711428, 1.2304309865340233}, .value = 4.2},
                                {.xyz = {-0.295233916417472, -0.098914913327117, 1.0586513091053404}, .value = 4.2},
                                {.xyz = {0.8108975722467402, 0.9557358148550632, 0.62609702872262}, .value = 4.2},
                                {.xyz = {1.022661511583011, 1.4500681465242986, 1.493478564105725}, .value = 4.2},
                                {.xyz = {-0.3955685158829483, -0.013308066198124413, 0.3765396965431256}, .value = 4.2},
                                {.xyz = {0.9694844761260912, -0.2214358103939862, 0.29815004698799474}, .value = 4.2},
                                {.xyz = {0.6285164206423317, 0.5761977623054122, 0.7831184875214461}, .value = 10.0},
                                {.xyz = {0.164468267614724, 0.19892673008276351, 0.15057765948320312}, .value = 10.0},
                                {.xyz = {0.6598929820547834, 0.8788011843084897, 0.8905499365323344}, .value = 10.0},
                                {.xyz = {0.5446957642369934, 0.5717283541076554, 0.4063456918612811}, .value = 10.0},
                                {.xyz = {0.20899393064792404, 0.23612888446153657, 0.16573066922880408}, .value = 10.0},
                                {.xyz = {0.5640572661463161, 0.8236832351082828, 0.6464808210801034}, .value = 10.0},
                                {.xyz = {0.585605705983617, 0.6513564103164575, 0.47986252327223977}, .value = 10.0},
                                {.xyz = {0.8086608285835175, 0.853825478118579, 0.63914145376668}, .value = 10.0},
                                {.xyz = {0.868582745661924, 0.9269701099236749, 1.122891859917919}, .value = 10.0}}},
        (GeometryTestScalarParameters){
            .createGeom =
                []() {
                    return std::make_shared<Union>(
                        std::vector<std::shared_ptr<ablate::mathFunctions::geom::Geometry>>{std::make_shared<Box>(std::vector<double>{1.0, 1.0}, std::vector<double>{2.0, 2.0}),
                                                                                            std::make_shared<Box>(std::vector<double>{1.5, 1.5}, std::vector<double>{2.5, 2.5})},
                        ablate::mathFunctions::Create(10.0),
                        ablate::mathFunctions::Create(-10.0));
                },
            .expectedResults = {{.xyz = {.25, .15}, .value = -10}, {.xyz = {1.75, 1.75}, .value = 10}, {.xyz = {2.25, 2.25}, .value = 10}, {.xyz = {2.25, 1.25}, .value = -10}}},
        (GeometryTestScalarParameters){
            .createGeom =
                []() {
                    return std::make_shared<Difference>(std::make_shared<Box>(std::vector<double>{1.0, 1.0}, std::vector<double>{2.0, 2.0}),
                                                        std::make_shared<Box>(std::vector<double>{1.5, 1.5}, std::vector<double>{2.5, 2.5}),
                                                        ablate::mathFunctions::Create(10.0),
                                                        ablate::mathFunctions::Create(-10.0));
                },
            .expectedResults = {{.xyz = {1.25, 1.25}, .value = 10}, {.xyz = {1.75, 1.75}, .value = -10}, {.xyz = {2.25, 2.25}, .value = -10}, {.xyz = {2.25, 1.25}, .value = -10}}},
        (GeometryTestScalarParameters){
            .createGeom =
                []() {
                    return std::make_shared<Inverse>(std::make_shared<Sphere>(std::vector<double>{0.0, 0.0, 1.99}, 2, ablate::mathFunctions::Create("20"), ablate::mathFunctions::Create("4.2")));
                },
            .expectedResults = {{.xyz = {0.0, 0.0, 2.0}, .value = 4.2}, {.xyz = {0.0, 0.0, 4.0}, .value = 20}}},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Triangle>(std::vector<double>{.5, .5, .5},
                                                                                 std::vector<double>{.6, .6, .9},
                                                                                 std::vector<double>{.9, .7, .54},
                                                                                 0.0,
                                                                                 ablate::mathFunctions::Create("20"),
                                                                                 ablate::mathFunctions::Create("4.2"));
                                           },
                                       .expectedResults = {{.xyz = {0.5, 0.5, 0.5}, .value = 20},
                                                           {.xyz = {.6, .6, .9}, .value = 20},
                                                           {.xyz = {.9, .7, .54}, .value = 20},
                                                           {.xyz = {(0.5 + .6) / 2.0, (0.5 + .6) / 2.0, (0.5 + .9) / 2.0}, .value = 20},
                                                           {.xyz = {(0.5 + .9) / 2.0, (0.5 + .7) / 2.0, (0.5 + .54) / 2.0}, .value = 20},
                                                           {.xyz = {(0.9 + .6) / 2.0, (0.7 + .6) / 2.0, (0.54 + .9) / 2.0}, .value = 20},
                                                           {.xyz = {0.4631011004118243, 0.5116781250263325, 0.7313051936404608}, .value = 4.2},
                                                           {.xyz = {0.41959218529309716, 0.6009858981647723, 0.7198554791355326}, .value = 4.2},
                                                           {.xyz = {0.5977408759813214, 0.6686988707982635, 0.6583900633051037}, .value = 20},
                                                           {.xyz = {0.15830083328217776, 1.570707379496506, 0.5427479468053291}, .value = 20},
                                                           {.xyz = {1.1154969658941738, -0.39406362954917085, 0.7946416659137492}, .value = 20},
                                                           {.xyz = {0.6760569231950302, 0.5079448791490717, 0.6789995494139746}, .value = 20},
                                                           {.xyz = {0.7522143005908749, 0.5132988700112692, 0.431362644034905}, .value = 4.2},
                                                           {.xyz = {0.7261017466228338, 0.525947138339539, 0.6292468420739659}, .value = 20},
                                                           {.xyz = {0.647785699409125, 0.6867011299887308, 0.608637355965095}, .value = 20},
                                                           {.xyz = {0.6390839163853796, 0.7045626846164187, 0.6063474130641094}, .value = 20},
                                                           {.xyz = {0.7348035296465791, 0.5080855837118511, 0.6315367849749515}, .value = 20},
                                                           {.xyz = {0.7331518961558077, 0.5119044352670835, 0.6676000343706964}, .value = 20},
                                                           {.xyz = {0.7244501131320623, 0.5297659898947714, 0.6653100914697108}, .value = 20},
                                                           {.xyz = {0.8147079104747921, 0.6898570142806325, 0.7849946515847246}, .value = 4.2},
                                                           {.xyz = {0.8147079104747921, 0.6898570142806325, 0.7849946515847246}, .value = 4.2},
                                                           {.xyz = {0.6852920895252079, 0.6101429857193673, 0.6550053484152754}, .value = 20},
                                                           {.xyz = {0.7244501131320623, 0.5297659898947714, 0.6653100914697108}, .value = 20},
                                                           {.xyz = {0.7331518961558077, 0.5119044352670835, 0.6676000343706964}, .value = 20},
                                                           {.xyz = {0.6374322828946082, 0.7083815361716511, 0.6424106624598543}, .value = 20}

                                       }},
        (GeometryTestScalarParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Triangle>(std::vector<double>{.5, .5, .5},
                                                                                 std::vector<double>{.6, .6, .9},
                                                                                 std::vector<double>{.9, .7, .54},
                                                                                 .1,
                                                                                 ablate::mathFunctions::Create("20"),
                                                                                 ablate::mathFunctions::Create("4.2"));
                                           },
                                       .expectedResults = {{.xyz = {0.5, 0.5, 0.5}, .value = 20},
                                                           {.xyz = {.6, .6, .9}, .value = 20},
                                                           {.xyz = {.9, .7, .54}, .value = 20},
                                                           {.xyz = {(0.5 + .6) / 2.0, (0.5 + .6) / 2.0, (0.5 + .9) / 2.0}, .value = 20},
                                                           {.xyz = {(0.5 + .9) / 2.0, (0.5 + .7) / 2.0, (0.5 + .54) / 2.0}, .value = 20},
                                                           {.xyz = {(0.9 + .6) / 2.0, (0.7 + .6) / 2.0, (0.54 + .9) / 2.0}, .value = 20},
                                                           {.xyz = {0.4631011004118243, 0.5116781250263325, 0.7313051936404608}, .value = 4.2},
                                                           {.xyz = {0.41959218529309716, 0.6009858981647723, 0.7198554791355326}, .value = 4.2},
                                                           {.xyz = {0.5977408759813214, 0.6686988707982635, 0.6583900633051037}, .value = 20},
                                                           {.xyz = {0.15830083328217776, 1.570707379496506, 0.5427479468053291}, .value = 4.2},
                                                           {.xyz = {1.1154969658941738, -0.39406362954917085, 0.7946416659137492}, .value = 4.2},
                                                           {.xyz = {0.6760569231950302, 0.5079448791490717, 0.6789995494139746}, .value = 20},
                                                           {.xyz = {0.7522143005908749, 0.5132988700112692, 0.431362644034905}, .value = 4.2},
                                                           {.xyz = {0.7261017466228338, 0.525947138339539, 0.6292468420739659}, .value = 20},
                                                           {.xyz = {0.647785699409125, 0.6867011299887308, 0.608637355965095}, .value = 20},
                                                           {.xyz = {0.6390839163853796, 0.7045626846164187, 0.6063474130641094}, .value = 4.2},
                                                           {.xyz = {0.7348035296465791, 0.5080855837118511, 0.6315367849749515}, .value = 4.2},
                                                           {.xyz = {0.7331518961558077, 0.5119044352670835, 0.6676000343706964}, .value = 4.2},
                                                           {.xyz = {0.7244501131320623, 0.5297659898947714, 0.6653100914697108}, .value = 20},
                                                           {.xyz = {0.8538659340816465, 0.6094800184560366, 0.79529939463916}, .value = 4.2},
                                                           {.xyz = {0.8147079104747921, 0.6898570142806325, 0.7849946515847246}, .value = 4.2},
                                                           {.xyz = {0.6852920895252079, 0.6101429857193673, 0.6550053484152754}, .value = 20},
                                                           {.xyz = {0.7244501131320623, 0.5297659898947714, 0.6653100914697108}, .value = 20},
                                                           {.xyz = {0.7331518961558077, 0.5119044352670835, 0.6676000343706964}, .value = 4.2},
                                                           {.xyz = {0.6374322828946082, 0.7083815361716511, 0.6424106624598543}, .value = 4.2}}}));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct GeometryTestVectorParameters {
    std::function<std::shared_ptr<Geometry>()> createGeom;
    std::vector<ExpectedValue<std::vector<double>>> expectedResults;
};

class GeometryTestVectorFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<GeometryTestVectorParameters> {};

TEST_P(GeometryTestVectorFixture, ShouldComputeCorrectAnswerFromXYZ) {
    // arrange
    const auto& param = GetParam();
    auto function = param.createGeom();
    for (const auto& [xyz, expectedValue] : param.expectedResults) {
        std::vector<double> result(expectedValue.size(), NAN);

        double x = xyz[0];
        double y = xyz.size() > 1 ? xyz[1] : NAN;
        double z = xyz.size() > 2 ? xyz[2] : NAN;

        // act
        function->Eval(x, y, z, NAN, result);

        // assert
        for (std::size_t i = 0; i < expectedValue.size(); i++) {
            ASSERT_DOUBLE_EQ(expectedValue[i], result[i]);
        }
    }
}

TEST_P(GeometryTestVectorFixture, ShouldComputeCorrectAnswerFromCoord) {
    // arrange
    const auto& param = GetParam();
    auto function = param.createGeom();
    for (const auto& [xyz, expectedValue] : param.expectedResults) {
        std::vector<double> result(expectedValue.size(), NAN);

        // act
        function->Eval(xyz.data(), (PetscInt)xyz.size(), NAN, result);

        // assert
        for (std::size_t i = 0; i < expectedValue.size(); i++) {
            ASSERT_DOUBLE_EQ(expectedValue[i], result[i]);
        }
    }
}

TEST_P(GeometryTestVectorFixture, ShouldComputeCorrectAnswerPetscFunction) {
    // arrange
    const auto& param = GetParam();
    auto function = param.createGeom();

    for (const auto& [xyz, expectedValue] : param.expectedResults) {
        std::vector<double> result(expectedValue.size(), NAN);

        auto context = function->GetContext();
        auto functionPointer = function->GetPetscFunction();

        // act
        auto errorCode = functionPointer((PetscInt)xyz.size(), NAN, xyz.data(), result.size(), &result[0], context);

        // assert
        ASSERT_EQ(errorCode, 0);
        for (std::size_t i = 0; i < expectedValue.size(); i++) {
            ASSERT_DOUBLE_EQ(expectedValue[i], result[i]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    GeometryTests, GeometryTestVectorFixture,
    testing::Values(
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{.2, .2}, .1, ablate::mathFunctions::Create(10.0));
                                           },
                                       .expectedResults = {{.xyz = {.25, .25}, .value = {10}}}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{.2, .2}, .1, ablate::mathFunctions::Create("12, 13, 14"));
                                           },
                                       .expectedResults = {{.xyz = {.3, .3}, .value = {0.0, 0.0, 0.0}}}},
        (GeometryTestVectorParameters){.createGeom = []() { return std::make_shared<Sphere>(std::vector<double>{0.0}, .1, ablate::mathFunctions::Create("12, 13, 14")); },
                                       .expectedResults = {{.xyz = {.1}, .value = {12.0, 13.0, 14.0}}}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Sphere>(std::vector<double>{-1, -2, -3}, 2, ablate::mathFunctions::Create("10, 10"), ablate::mathFunctions::Create("4.2, 6.2"));
                                           },
                                       .expectedResults = {{.xyz = {10, 11, 12}, .value = {4.2, 6.2}}}},
        (GeometryTestVectorParameters){
            .createGeom =
                []() {
                    return std::make_shared<Sphere>(std::vector<double>{0.0, 0.0, 1.99}, 2, ablate::mathFunctions::Create("20, 13"), ablate::mathFunctions::Create("4.2, 4.2"));
                },
            .expectedResults = {{.xyz = {0.0, 0.0, 2.0}, .value = {20, 13}}}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{.2, .1}, std::vector<double>{.3, .2}, ablate::mathFunctions::Create(std::vector<double>{10, 13}));
                                           },
                                       .expectedResults = {{.xyz = {.25, .15}, .value = {10, 13}}}},
        (GeometryTestVectorParameters){.createGeom = []() { return std::make_shared<Box>(std::vector<double>{.2}, std::vector<double>{.3}, ablate::mathFunctions::Create(10.0)); },
                                       .expectedResults = {{.xyz = {.15}, .value = {0}}}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{.2, .1, -.1},
                                                                            std::vector<double>{.3, .2, .2},
                                                                            ablate::mathFunctions::Create(std::vector<double>{10, 10}),
                                                                            ablate::mathFunctions::Create(std::vector<double>{4.2, .23}));
                                           },
                                       .expectedResults = {{.xyz = {.25, .25, .25}, .value = {4.2, .23}}}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{-2, -2},
                                                                            std::vector<double>{-1, -1},
                                                                            ablate::mathFunctions::Create(std::vector<double>{10, 11}),
                                                                            ablate::mathFunctions::Create(std::vector<double>{4.2, 4.2}));
                                           },
                                       .expectedResults = {{.xyz = {-1.5, -1.0000001}, .value = {10, 11}}}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{-2, -2}, std::vector<double>{-1, -1}, ablate::mathFunctions::Create(std::vector<double>{1, 2}));
                                           },
                                       .expectedResults = {{.xyz = {-1.5, -1.0000001, 2}, .value = {1, 2}}}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Box>(std::vector<double>{-2, -2},
                                                                            std::vector<double>{-1, -1},
                                                                            ablate::mathFunctions::Create(std::vector<double>{3, 3, 3, 3}),
                                                                            ablate::mathFunctions::Create(std::vector<double>{0, 1, 2, 3}));
                                           },
                                       .expectedResults = {{.xyz = {-1.5, -2.0000001, 2}, .value = {0, 1, 2, 3}}}},
        (GeometryTestVectorParameters){.createGeom = []() { return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step"); },
                                       .expectedResults = {{.xyz = {0.0, 0.0, 0.0}, .value = {1}}}},
        (GeometryTestVectorParameters){.createGeom = []() { return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step"); },
                                       .expectedResults = {{.xyz = {0.0, 0.041, 0.0}, /**41 mm should be outside ***/
                                                            .value = {0}}}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step",
                                                                                ablate::mathFunctions::Create(std::vector<double>{10, 23, 22, 11}),
                                                                                ablate::mathFunctions::Create(std::vector<double>{0, 1, 2, 3}));
                                           },
                                       .expectedResults = {{.xyz = {0.005, 0.035, 0.005}, .value = {10, 23, 22, 11}}}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Surface>("inputs/mathFunctions/geom/testShape_m.step",
                                                                                ablate::mathFunctions::Create(std::vector<double>{4.2, 0.0}),
                                                                                ablate::mathFunctions::Create(std::vector<double>{1, 2}));
                                           },
                                       .expectedResults = {{.xyz = {0.01, 0.035, 0.01}, .value = {1, 2}}}},

        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<CylinderShell>(std::vector<double>{-0.001, -0.25, 0.1519}, std::vector<double>{-0.001, -0.23, 0.1519}, .1, .2);
                                           },
                                       .expectedResults = {{.xyz = {0.09353390997023181, -0.237586838224223, 0.10392668643347691}, .value = {1}},
                                                           {.xyz = {0.16784382726924696, -0.24078984395323688, 0.16180769506641768}, .value = {1}},
                                                           {.xyz = {0.06795544842480095, -0.24119896516384987, 0.2405161434782157}, .value = {1}},
                                                           {.xyz = {-0.10240145684754887, -0.2347519315508647, 0.19731202506093004}, .value = {1}},
                                                           {.xyz = {-0.13065410514914433, -0.2466488199479725, 0.3032415139585555}, .value = {1}},
                                                           {.xyz = {0.11338404742910957, -0.24287313833346336, 0.11304931383706829}, .value = {1}},
                                                           {.xyz = {-0.07294125536895812, -0.23408284773497967, 0.005703529030037902}, .value = {1}},
                                                           {.xyz = {0.11328869080553239, -0.2425755986512529, 0.28721084324462876}, .value = {1}},
                                                           {.xyz = {0.1049742940853976, -0.2410616559964388, 0.13651048929841914}, .value = {1}},
                                                           {.xyz = {0.0166443880945093, -0.2492774851386651, 0.27864843973915954}, .value = {1}},
                                                           {.xyz = {0.05694208394754052, -0.23842202869792728, 0.25339822636736375}, .value = {1}},
                                                           {.xyz = {-0.037291395450233966, -0.24857316186017686, 0.03008670922077772}, .value = {1}},
                                                           {.xyz = {0.1771439161925853, -0.2486900482437031, 0.15259685599599748}, .value = {1}},
                                                           {.xyz = {0.10738635835047228, -0.24139873210027296, 0.04586928741299445}, .value = {1}},
                                                           {.xyz = {0.07156403055912952, -0.15136173333868808, -0.34593954910019187}, .value = {0}},
                                                           {.xyz = {-0.17995889737037196, -0.3357635106149338, 0.13472779934552137}, .value = {0}},
                                                           {.xyz = {0.39647886810208643, -0.15777978229062772, 0.29838655984614193}, .value = {0}},
                                                           {.xyz = {0.2065338084143079, -0.23247509509848452, 0.17085116726778693}, .value = {0}},
                                                           {.xyz = {-0.3939149905035768, 0.4840345739239337, 0.3702902271464009}, .value = {0}},
                                                           {.xyz = {0.17141584334559434, -0.1878204350497401, -0.4657417783545752}, .value = {0}},
                                                           {.xyz = {0.3266427542923126, -0.011672888093754663, 0.21694519404266266}, .value = {0}},
                                                           {.xyz = {-0.4781415969674002, -0.2512462056016449, 0.35222851934245525}, .value = {0}},
                                                           {.xyz = {0.3377701393009437, -0.010215826152739194, -0.08796550800916347}, .value = {0}},
                                                           {.xyz = {-0.3568148493635891, -5.2852579357720586E-5, 0.005475293073376131}, .value = {0}}}},
        (GeometryTestVectorParameters){
            .createGeom =
                []() {
                    return std::make_shared<CylinderShell>(
                        std::vector<double>{1, 1, 1}, std::vector<double>{0.0, 0.0, 0.0}, .1, .2, ablate::mathFunctions::Create(10.0), ablate::mathFunctions::Create(4.2));
                },
            .expectedResults =
                {{.xyz = {0.5737255071346832, 0.767022803088282, 0.5894503202916701}, .value = {10.0}},       {.xyz = {0.9350312671904069, 0.8860091649029023, 1.0366283632890532}, .value = {10.0}},
                 {.xyz = {0.028457437231878435, -0.10758872029407462, 0.10596322836141181}, .value = {10.0}}, {.xyz = {0.6882172726971765, 0.5592666398031048, 0.7046186970763693}, .value = {10.0}},
                 {.xyz = {0.5666051698326102, 0.7263407208655104, 0.7367112249878809}, .value = {10.0}},      {.xyz = {0.511871914214413, 0.4666553472974948, 0.7018657959693819}, .value = {10.0}},
                 {.xyz = {0.3804938917388385, 0.41052942969170236, 0.23002427367601164}, .value = {10.0}},    {.xyz = {0.8033101569961627, 0.9821200990912846, 0.9141486979760338}, .value = {10.0}},
                 {.xyz = {0.5503625040357609, 0.7242286828645175, 0.6078857913255158}, .value = {10.0}},      {.xyz = {0.4680779770846777, 0.27561050343341265, 0.5031436582795212}, .value = {10.0}},
                 {.xyz = {0.03784584080408915, 0.7347327767365763, 0.08749700095536728}, .value = {4.2}},     {.xyz = {0.8988149395062159, -0.03626073415056208, 0.8340999023803175}, .value = {4.2}},
                 {.xyz = {1.4866610476053264, 0.8405522155957352, -0.32938707732243633}, .value = {4.2}},     {.xyz = {-0.3579880975123806, 1.1685083809388985, 0.7102663693480407}, .value = {4.2}},
                 {.xyz = {1.4042265778723018, -0.43335570937281354, 1.4878122769063051}, .value = {4.2}},     {.xyz = {0.4273613902405924, -0.08227590932321416, -0.4809993582520806}, .value = {4.2}},
                 {.xyz = {0.8785875082590098, -0.39824072507040564, 0.9249726070610769}, .value = {4.2}},     {.xyz = {0.4367722443348041, 1.316090072841381, 1.1375032842530228}, .value = {4.2}},
                 {.xyz = {1.0786837986186593, 0.35271030319116625, -0.09724984881334531}, .value = {4.2}},    {.xyz = {1.1199175152976288, -0.3747659414706568, -0.13500782662299615}, .value = {4.2}},
                 {.xyz = {-0.2300050335134063, -0.452093404799514, 0.35880262351180936}, .value = {4.2}}}},
        (GeometryTestVectorParameters){
            .createGeom =
                []() {
                    return std::make_shared<Cylinder>(std::vector<double>{1, 1, 1}, std::vector<double>{0.0, 0.0, 0.0}, .2, ablate::mathFunctions::Create(10.0), ablate::mathFunctions::Create(4.2));
                },
            .expectedResults = {{.xyz = {.5, .5, .5}, .value = {10.0}},
                                {.xyz = {0.3469498160486115, 0.12212685458885075, 0.5935674027949154}, .value = {4.2}},
                                {.xyz = {1.422497529062558, 0.0747921342509037, 0.5438022732511887}, .value = {4.2}},
                                {.xyz = {-0.3974364402389252, 7.580389995769377E-4, 1.0031405621614435}, .value = {4.2}},
                                {.xyz = {0.6201884301886198, -0.38980652133015936, 1.361877392308243}, .value = {4.2}},
                                {.xyz = {0.5387681888518165, -0.14883385878364908, -0.05712957242873018}, .value = {4.2}},
                                {.xyz = {-0.053761378342362276, 0.905409062711428, 1.2304309865340233}, .value = {4.2}},
                                {.xyz = {-0.295233916417472, -0.098914913327117, 1.0586513091053404}, .value = {4.2}},
                                {.xyz = {0.8108975722467402, 0.9557358148550632, 0.62609702872262}, .value = {4.2}},
                                {.xyz = {1.022661511583011, 1.4500681465242986, 1.493478564105725}, .value = {4.2}},
                                {.xyz = {-0.3955685158829483, -0.013308066198124413, 0.3765396965431256}, .value = {4.2}},
                                {.xyz = {0.9694844761260912, -0.2214358103939862, 0.29815004698799474}, .value = {4.2}},
                                {.xyz = {0.6285164206423317, 0.5761977623054122, 0.7831184875214461}, .value = {10.0}},
                                {.xyz = {0.164468267614724, 0.19892673008276351, 0.15057765948320312}, .value = {10.0}},
                                {.xyz = {0.6598929820547834, 0.8788011843084897, 0.8905499365323344}, .value = {10.0}},
                                {.xyz = {0.5446957642369934, 0.5717283541076554, 0.4063456918612811}, .value = {10.0}},
                                {.xyz = {0.20899393064792404, 0.23612888446153657, 0.16573066922880408}, .value = {10.0}},
                                {.xyz = {0.5640572661463161, 0.8236832351082828, 0.6464808210801034}, .value = {10.0}},
                                {.xyz = {0.585605705983617, 0.6513564103164575, 0.47986252327223977}, .value = {10.0}},
                                {.xyz = {0.8086608285835175, 0.853825478118579, 0.63914145376668}, .value = {10.0}},
                                {.xyz = {0.868582745661924, 0.9269701099236749, 1.122891859917919}, .value = {10.0}}}},
        (GeometryTestVectorParameters){
            .createGeom =
                []() {
                    return std::make_shared<Union>(
                        std::vector<std::shared_ptr<ablate::mathFunctions::geom::Geometry>>{std::make_shared<Box>(std::vector<double>{1.0, 1.0}, std::vector<double>{2.0, 2.0}),
                                                                                            std::make_shared<Box>(std::vector<double>{1.5, 1.5}, std::vector<double>{2.5, 2.5})},
                        ablate::mathFunctions::Create(10.0),
                        ablate::mathFunctions::Create(-10.0));
                },
            .expectedResults = {{.xyz = {.25, .15}, .value = {-10}}, {.xyz = {1.75, 1.75}, .value = {10}}, {.xyz = {2.25, 2.25}, .value = {10}}, {.xyz = {2.25, 1.25}, .value = {-10}}}},
        (GeometryTestVectorParameters){
            .createGeom =
                []() {
                    return std::make_shared<Difference>(std::make_shared<Box>(std::vector<double>{1.0, 1.0}, std::vector<double>{2.0, 2.0}),
                                                        std::make_shared<Box>(std::vector<double>{1.5, 1.5}, std::vector<double>{2.5, 2.5}),
                                                        ablate::mathFunctions::Create(10.0),
                                                        ablate::mathFunctions::Create(-10.0));
                },
            .expectedResults = {{.xyz = {1.25, 1.25}, .value = {10}}, {.xyz = {1.75, 1.75}, .value = {-10}}, {.xyz = {2.25, 2.25}, .value = {-10}}, {.xyz = {2.25, 1.25}, .value = {-10}}}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Inverse>(std::make_shared<Sphere>(
                                                   std::vector<double>{0.0, 0.0, 1.99}, 2, ablate::mathFunctions::Create("20, 13"), ablate::mathFunctions::Create("4.2, 4.2")));
                                           },
                                       .expectedResults = {{.xyz = {0.0, 0.0, 2.0}, .value = {4.2, 4.2}}, {.xyz = {0.0, 0.0, 4.0}, .value = {20, 13}}}},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Triangle>(std::vector<double>{.5, .5, .5},
                                                                                 std::vector<double>{.6, .6, .9},
                                                                                 std::vector<double>{.9, .7, .54},
                                                                                 0.0,
                                                                                 ablate::mathFunctions::Create(std::vector<double>{20, 11}),
                                                                                 ablate::mathFunctions::Create(std::vector<double>{4.2, -10}));
                                           },
                                       .expectedResults = {{.xyz = {0.5, 0.5, 0.5}, .value = {20, 11}},
                                                           {.xyz = {.6, .6, .9}, .value = {20, 11}},
                                                           {.xyz = {.9, .7, .54}, .value = {20, 11}},
                                                           {.xyz = {(0.5 + .6) / 2.0, (0.5 + .6) / 2.0, (0.5 + .9) / 2.0}, .value = {20, 11}},
                                                           {.xyz = {(0.5 + .9) / 2.0, (0.5 + .7) / 2.0, (0.5 + .54) / 2.0}, .value = {20, 11}},
                                                           {.xyz = {(0.9 + .6) / 2.0, (0.7 + .6) / 2.0, (0.54 + .9) / 2.0}, .value = {20, 11}},
                                                           {.xyz = {0.4631011004118243, 0.5116781250263325, 0.7313051936404608}, .value = {4.2, -10}},
                                                           {.xyz = {0.41959218529309716, 0.6009858981647723, 0.7198554791355326}, .value = {4.2, -10}},
                                                           {.xyz = {0.5977408759813214, 0.6686988707982635, 0.6583900633051037}, .value = {20, 11}},
                                                           {.xyz = {0.15830083328217776, 1.570707379496506, 0.5427479468053291}, .value = {20, 11}},
                                                           {.xyz = {1.1154969658941738, -0.39406362954917085, 0.7946416659137492}, .value = {20, 11}},
                                                           {.xyz = {0.6760569231950302, 0.5079448791490717, 0.6789995494139746}, .value = {20, 11}},

                                                           {.xyz = {0.7522143005908749, 0.5132988700112692, 0.431362644034905}, .value = {4.2, -10}},
                                                           {.xyz = {0.7261017466228338, 0.525947138339539, 0.6292468420739659}, .value = {20, 11}},
                                                           {.xyz = {0.647785699409125, 0.6867011299887308, 0.608637355965095}, .value = {20, 11}},
                                                           {.xyz = {0.6390839163853796, 0.7045626846164187, 0.6063474130641094}, .value = {20, 11}},
                                                           {.xyz = {0.7348035296465791, 0.5080855837118511, 0.6315367849749515}, .value = {20, 11}},

                                                           {.xyz = {0.7331518961558077, 0.5119044352670835, 0.6676000343706964}, .value = {20, 11}},
                                                           {.xyz = {0.7244501131320623, 0.5297659898947714, 0.6653100914697108}, .value = {20, 11}},

                                                           {.xyz = {0.8147079104747921, 0.6898570142806325, 0.7849946515847246}, .value = {4.2, -10}},
                                                           {.xyz = {0.6852920895252079, 0.6101429857193673, 0.6550053484152754}, .value = {20, 11}},
                                                           {.xyz = {0.7244501131320623, 0.5297659898947714, 0.6653100914697108}, .value = {20, 11}},
                                                           {.xyz = {0.7331518961558077, 0.5119044352670835, 0.6676000343706964}, .value = {20, 11}},
                                                           {.xyz = {0.6374322828946082, 0.7083815361716511, 0.6424106624598543}, .value = {20, 11}}

                                       }},
        (GeometryTestVectorParameters){.createGeom =
                                           []() {
                                               return std::make_shared<Triangle>(std::vector<double>{.5, .5, .5},
                                                                                 std::vector<double>{.6, .6, .9},
                                                                                 std::vector<double>{.9, .7, .54},
                                                                                 .1,
                                                                                 ablate::mathFunctions::Create(std::vector<double>{20, 11}),
                                                                                 ablate::mathFunctions::Create(std::vector<double>{4.2, -10}));
                                           },
                                       .expectedResults = {{.xyz = {0.5, 0.5, 0.5}, .value = {20, 11}},
                                                           {.xyz = {.6, .6, .9}, .value = {20, 11}},
                                                           {.xyz = {.9, .7, .54}, .value = {20, 11}},
                                                           {.xyz = {(0.5 + .6) / 2.0, (0.5 + .6) / 2.0, (0.5 + .9) / 2.0}, .value = {20, 11}},
                                                           {.xyz = {(0.5 + .9) / 2.0, (0.5 + .7) / 2.0, (0.5 + .54) / 2.0}, .value = {20, 11}},
                                                           {.xyz = {(0.9 + .6) / 2.0, (0.7 + .6) / 2.0, (0.54 + .9) / 2.0}, .value = {20, 11}},
                                                           {.xyz = {0.4631011004118243, 0.5116781250263325, 0.7313051936404608}, .value = {4.2, -10}},
                                                           {.xyz = {0.41959218529309716, 0.6009858981647723, 0.7198554791355326}, .value = {4.2, -10}},
                                                           {.xyz = {0.5977408759813214, 0.6686988707982635, 0.6583900633051037}, .value = {20, 11}},
                                                           {.xyz = {0.15830083328217776, 1.570707379496506, 0.5427479468053291}, .value = {4.2, -10}},
                                                           {.xyz = {1.1154969658941738, -0.39406362954917085, 0.7946416659137492}, .value = {4.2, -10}},
                                                           {.xyz = {0.6760569231950302, 0.5079448791490717, 0.6789995494139746}, .value = {20, 11}},
                                                           {.xyz = {0.7522143005908749, 0.5132988700112692, 0.431362644034905}, .value = {4.2, -10}},
                                                           {.xyz = {0.7261017466228338, 0.525947138339539, 0.6292468420739659}, .value = {20, 11}},
                                                           {.xyz = {0.647785699409125, 0.6867011299887308, 0.608637355965095}, .value = {20, 11}},
                                                           {.xyz = {0.6390839163853796, 0.7045626846164187, 0.6063474130641094}, .value = {4.2, -10}},
                                                           {.xyz = {0.7348035296465791, 0.5080855837118511, 0.6315367849749515}, .value = {4.2, -10}},
                                                           {.xyz = {0.7331518961558077, 0.5119044352670835, 0.6676000343706964}, .value = {4.2, -10}},
                                                           {.xyz = {0.7244501131320623, 0.5297659898947714, 0.6653100914697108}, .value = {20, 11}},
                                                           {.xyz = {0.8538659340816465, 0.6094800184560366, 0.79529939463916}, .value = {4.2, -10}},
                                                           {.xyz = {0.6852920895252079, 0.6101429857193673, 0.6550053484152754}, .value = {20, 11}},
                                                           {.xyz = {0.7244501131320623, 0.5297659898947714, 0.6653100914697108}, .value = {20, 11}},
                                                           {.xyz = {0.7331518961558077, 0.5119044352670835, 0.6676000343706964}, .value = {4.2, -10}},
                                                           {.xyz = {0.6374322828946082, 0.7083815361716511, 0.6424106624598543}, .value = {4.2, -10}}}}

        ));

}  // namespace ablateTesting::mathFunctions::geom