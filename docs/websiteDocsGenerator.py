import argparse
import pathlib
import re
from dataclasses import dataclass

import inflection

# define some required strings
SimulationsTitle = "Running Simulations"
ComponentsTitle = "Component List"
ExamplesTitle = "Examples Input Files"
ExamplesIndex = "_exampleList.md"
BaseExampleUrl = "https://github.com/UBCHREST/ablate/tree/main/tests/integrationTests/inputs/"


def format_title(title):
    return title

def print_interface_file(interface_name, component_output_dir, interface_documentation):
    # write the buffer to the file
    component_file_path = component_output_dir / (interface_name + ".md")
    with open(component_file_path, 'w') as component_file:
        # write the header
        component_file.write('---\n')
        component_file.write('layout: default\n')
        component_file.write(f'title: {interface_name}\n')
        component_file.write(f'parent: {ComponentsTitle}\n')
        component_file.write(f'grand_parent: {SimulationsTitle}\n')
        component_file.write('nav_exclude: true\n')
        component_file.write('---\n')

        component_file.writelines(interface_documentation)


# Convert the single markdown component list into multiple files
def split_component_list(component_list_source, component_output_dir):
    if component_output_dir is None:
        component_output_dir = component_list_source.parent / 'output'
    component_output_dir.mkdir(parents=True, exist_ok=True)

    # parse once to create a list of known interfaces
    known_interfaces = set()
    with open(component_list_source, 'r') as text_file:
        for row in text_file:
            if row.startswith('# '):
                # set the interface_name
                known_interfaces.add(format_title(row[2:-1]))

    # separate each file
    interface_name = ''
    interface_documentation = []

    # march over each line
    with open(component_list_source, 'r') as text_file:
        for row in text_file:
            if row.startswith('# ') and interface_documentation:
                # write the buffer to the file
                print_interface_file(interface_name, component_output_dir, interface_documentation)
                # clear the buffer
                interface_documentation.clear()

            if row.startswith('# '):
                # set the interface_name
                interface_name = format_title(row[2:-1])

            # check the row for known interfaces
            match_results = re.findall(r'(ablate::.*?)(,|\s|\))', row)
            for match in match_results:
                test_name = match[0]
                if test_name in known_interfaces:
                    row = row.replace(test_name, f'[{test_name}](./{test_name}.html)')

            interface_documentation.append(row)

        # print the last item
        if interface_documentation:
            print_interface_file(interface_name, component_output_dir, interface_documentation)

@dataclass
class ExampleMetaData:
    title: str
    doc_url: str
    file_url: str


# Convert each yaml input into a markdown example for the website
def create_example_files(example_input_directory, example_output_directory):
    if example_output_directory is None:
        example_output_directory = example_input_directory.parent / 'output_examples'
    example_output_directory.mkdir(parents=True, exist_ok=True)

    # store a list of categories and examples
    categories_meta_data = dict()

    # march over each example directory for categories
    for example_directory in example_input_directory.iterdir():
        if example_directory.is_dir():
            example_category_title = inflection.humanize(inflection.underscore(example_directory.stem))

            # keep a list of example metas to build a table
            example_meta_datas = []

            # make sure that there are input files here
            yaml_files = example_directory.glob("*.yaml")
            if yaml_files:
                # create the directory
                output_category_directory = example_output_directory / example_directory.stem
                output_category_directory.mkdir(parents=True, exist_ok=True)

                for yaml_file in example_directory.glob("*.yaml"):
                    # compute the metadata
                    example_title = inflection.humanize(inflection.underscore(yaml_file.stem))
                    example_doc_url = f'./{example_output_directory.name}/{example_directory.name}/{yaml_file.stem}.html'
                    example_file_url = BaseExampleUrl + example_directory.name + "/" + yaml_file.name
                    example_meta_datas.append(ExampleMetaData(example_title, example_doc_url, example_file_url))

                    # copy over the input as markdown
                    header_region = True
                    with open(yaml_file, 'r') as yaml_file_input:
                        with open(output_category_directory / (yaml_file.stem + '.md'), 'w') as markdown_file:
                            # output the header information
                            markdown_file.write('---\n')
                            markdown_file.write('layout: default\n')
                            markdown_file.write(f'title: {example_title}\n')
                            markdown_file.write(f'parent: {ExamplesTitle}\n')
                            markdown_file.write(f'grand_parent: {SimulationsTitle}\n')
                            markdown_file.write('nav_exclude: true\n')
                            markdown_file.write('---\n')

                            # copy over each row
                            for row in yaml_file_input:
                                if header_region:
                                    # check to see if we are done with header region
                                    if row.startswith('---'):
                                        header_region = False
                                        markdown_file.write(
                                            f'\n[{example_directory.name + "/" + yaml_file.name}]({example_file_url})\n')
                                        markdown_file.write("```yaml\n")

                                if header_region:
                                    markdown_file.write(row.replace("#", "", 1).replace(" ", "", 1))
                                else:
                                    markdown_file.write(row)


                            # close off input
                            markdown_file.write("\n```")

        if example_meta_datas:
            categories_meta_data[example_category_title] = example_meta_datas
            #
            # # output an index file
            # with open(output_category_directory / 'index.md', 'w') as index_file:
            #
            #     # write the header
            #     index_file.write('---\n')
            #     index_file.write('layout: default\n')
            #     index_file.write(f'title: {example_category_title}\n')
            #     index_file.write(f'parent: {ExamplesTitle}\n')
            #     index_file.write(f'grand_parent: {SimulationsTitle}\n')
            #     index_file.write(f'has_children: true\n')
            #     index_file.write('---\n')

            # copy over each file

        # print(example_directory_title)

    # output an index file
    with open(example_output_directory / ExamplesIndex, 'w') as index_file:
        for category, meta_data_list in categories_meta_data.items():
            # write the header
            index_file.write(f'## {category}\n')
            for meta_data in meta_data_list:
                index_file.write(f'- [{meta_data.title}]({meta_data.doc_url}) [📁]({meta_data.file_url})\n')
            index_file.write('\n')


#
# # parse once to create a list of known interfaces
# known_interfaces = set()
# with open(component_list_source, 'r') as text_file:
#     for row in text_file:
#         if row.startswith('# '):
#             # set the interface_name
#             known_interfaces.add(format_title(row[2:-1]))
#
# # separate each file
# interface_name = ''
# interface_documentation = []
#
# # march over each line
# with open(component_list_source, 'r') as text_file:
#     for row in text_file:
#         if row.startswith('# ') and interface_documentation:
#             # write the buffer to the file
#             component_file_path = component_output_dir / (interface_name + ".md")
#             with open(component_file_path, 'w') as component_file:
#                 # write the header
#                 component_file.write('---\n')
#                 component_file.write('layout: default\n')
#                 component_file.write(f'title: {interface_name}\n')
#                 component_file.write('parent: Component List\n')
#                 component_file.write('grand_parent: Running Simulations\n')
#                 component_file.write('nav_exclude: true\n')
#                 component_file.write('---\n')
#
#                 component_file.writelines(interface_documentation)
#
#             # clear the buffer
#             interface_documentation.clear()
#
#         if row.startswith('# '):
#             # set the interface_name
#             interface_name = format_title(row[2:-1])
#
#         # check the row for known interfaces
#         match_results = re.findall(r'(ablate::.*?)(,|\s|\))', row)
#         for match in match_results:
#             test_name = match[0]
#             if test_name in known_interfaces:
#                 row = row.replace(test_name, f'[{test_name}](./{test_name}.html)')
#
#         interface_documentation.append(row)


# Main function to parser input files and run the document generator
def parse():
    parser = argparse.ArgumentParser(description='Generate additional documentation for ablate.dev')
    parser.add_argument('--componentListSource', dest='component_list_source', type=pathlib.Path,
                        help='The path to the markdown file with all components')
    parser.add_argument('--componentOutputDir', dest='component_output_dir', type=pathlib.Path,
                        help='Optional path to output directory for components')
    parser.add_argument('--exampleInputDir', dest='example_input_dir', type=pathlib.Path,
                        help='The directory holding the yaml input file examples')
    parser.add_argument('--exampleOutputDir', dest='example_output_dir', type=pathlib.Path,
                        help='Optional path to output directory for the generated example markdowns')
    args = parser.parse_args()

    # create component markdown
    if args.component_list_source:
        split_component_list(args.component_list_source, args.component_output_dir)
    if args.example_input_dir:
        create_example_files(args.example_input_dir, args.example_output_dir)


if __name__ == "__main__":
    parse()
