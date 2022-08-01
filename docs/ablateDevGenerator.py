import os
import subprocess
import componentListGenerator
import argparse
import pathlib
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate documentation for ablate.dev assuming that this is run from the root repo directory')
    parser.add_argument('--ablateExe', dest='ablate_exe', type=pathlib.Path,
                        help='The path to the ablate executable', required=True)
    parser.add_argument('--rootDir', dest='root_dir', default=os.getcwd(), type=pathlib.Path,
                        help='the root repo directory')
    args = parser.parse_args()

    # Get the version from the build file
    ablateVersion = subprocess.run([str(args.ablate_exe), "--version"], check=True, capture_output=True,
                                   text=True).stdout
    with open(args.root_dir / 'docs/_includes/generated/version.html', 'w') as f:
        print(ablateVersion, file=f, end='')

    # get the petsc build information
    completeInformation = subprocess.run([str(args.ablate_exe), "-version"], check=False, capture_output=True,
                                         text=True).stdout
    petscVersionMatches = re.search('revision: ([^\s]+)', completeInformation)
    with open(args.root_dir / 'docs/_includes/generated/petscVersion.html', 'w') as f:
        print(petscVersionMatches.group(1), file=f, end='')

    # Get the component information
    componentInformation = subprocess.run([str(args.ablate_exe), "--help"], check=False, capture_output=True,
                                          text=True).stdout
    componentInformationFile = args.root_dir / 'docs/_componentListSource.md'
    with open(str(componentInformationFile), 'w') as f:
        print(componentInformation, file=f, end='')

    # create component markdown
    componentListGenerator.split_component_list(componentInformationFile,
                                                args.root_dir / 'docs/content/simulations/components')

    # Create example file
    componentListGenerator.create_example_files(args.root_dir / 'tests/integrationTests/inputs',
                                                args.root_dir / 'docs/content/simulations/examples')

    # Update the doxyfile.config with the version
    doxyFileOrg = open(args.root_dir / 'docs/doxyfile.config', "r")
    with open(args.root_dir / 'docs/doxyfile.tmp.config', 'w') as doxyFile:
        doxyFile.write(doxyFileOrg.read())
        doxyFile.write(f'PROJECT_NUMBER = {ablateVersion}')

    # call doxygen
    completeInformation = subprocess.run(['doxygen', args.root_dir / 'docs/doxyfile.tmp.config'], check=True)

    # versionFile = args.doc_dir/'_includes/generated/version.html'
    # with open(versionFile, 'w') as f:
    #     print(ablateVersion, file=f, end='')
    #
    # # set the directory information
    # componentListSource = args.doc_dir / "_componentListSource.md"
    #

    # parser.add_argument('--componentListSource', dest='component_list_source', type=pathlib.Path,
    #                     help='The path to the markdown file with all components')
    # parser.add_argument('--componentOutputDir', dest='component_output_dir', type=pathlib.Path,
    #                     help='Optional path to output directory for components')
    # parser.add_argument('--exampleInputDir', dest='example_input_dir', type=pathlib.Path,
    #                     help='The directory holding the yaml input file examples')
    # parser.add_argument('--exampleOutputDir', dest='example_output_dir', type=pathlib.Path,
    #                     help='Optional path to output directory for the generated example markdowns')
    #
    # # create component markdown
    # if args.component_list_source:
    #     componentListGenerator.split_component_list(args.component_list_source, args.component_output_dir)
    # if args.example_input_dir:
    #     componentListGenerator.create_example_files(args.example_input_dir, args.example_output_dir)
