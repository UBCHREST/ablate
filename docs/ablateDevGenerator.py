import os
import subprocess
import yaml
import componentListGenerator
import argparse
import pathlib
import re
import configGenerator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate documentation for ablate.dev assuming that this is run from the root repo directory')
    parser.add_argument('--ablateExe', dest='ablate_exe', type=pathlib.Path,
                        help='The path to the ablate executable', required=True)
    parser.add_argument('--rootDir', dest='root_dir', default=os.getcwd(), type=pathlib.Path,
                        help='the root repo directory')
    args = parser.parse_args()

    # Get the version and other metadata from the build file
    ablate_metadata_string = subprocess.run([str(args.ablate_exe), "--info"], check=True, capture_output=True,
                                            text=True).stdout
    ablate_metadata = yaml.safe_load(ablate_metadata_string)['ABLATE']

    # write the version to a file
    with open(args.root_dir / 'docs/_includes/generated/version.html', 'w') as f:
        print(ablate_metadata['version'], file=f, end='')

    # write the version-commit to the petsc version file
    with open(args.root_dir / 'docs/_includes/generated/petscVersion.html', 'w') as f:
        print(f'{ablate_metadata["petscVersion"]} {ablate_metadata["petscGitCommit"]} ', file=f, end='')

    # generate any required config files
    # build the replacement dictionary
    configGenerator.generate_config_files(args.root_dir / 'config', args.root_dir / 'docs/config', ablate_metadata)

    # Get the component information
    componentInformation = subprocess.run([str(args.ablate_exe), "--help"], check=False, capture_output=True,
                                          text=True).stdout
    componentInformationFile = args.root_dir / 'docs/_componentListSource.md'
    with open(str(componentInformationFile), 'w') as f:
        print(componentInformation, file=f, end='')

    # create component markdown
    componentListGenerator.split_component_list(componentInformationFile,
                                                args.root_dir / 'docs/content/simulations/components')

    # Create example file lists
    componentListGenerator.create_example_files(args.root_dir / 'tests/integrationTests/inputs',
                                                args.root_dir / 'docs/content/simulations/integrationExamples')

    componentListGenerator.create_example_files(args.root_dir / 'tests/regressionTests/inputs',
                                                args.root_dir / 'docs/content/simulations/regressionExamples')

    # Update the doxyfile.config with the version
    doxyFileOrg = open(args.root_dir / 'docs/doxyfile.config', "r")
    with open(args.root_dir / 'docs/doxyfile.tmp.config', 'w') as doxyFile:
        doxyFile.write(doxyFileOrg.read())
        doxyFile.write(f'PROJECT_NUMBER = {ablate_metadata["version"]}')

    # call doxygen
    completeInformation = subprocess.run(['doxygen', args.root_dir / 'docs/doxyfile.tmp.config'], cwd=args.root_dir,
                                         check=True)
