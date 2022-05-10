import argparse
import pathlib
import re


def format_title(title):
    return title


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
                component_file_path = component_output_dir / (interface_name + ".md")
                with open(component_file_path, 'w') as component_file:
                    # write the header
                    component_file.write('---\n')
                    component_file.write('layout: default\n')
                    component_file.write(f'title: {interface_name}\n')
                    component_file.write('parent: Component List\n')
                    component_file.write('grand_parent: Running Simulations\n')
                    component_file.write('nav_exclude: true\n')
                    component_file.write('---\n')

                    component_file.writelines(interface_documentation)

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


# Main function to parser input files and run the document generator
def parse():
    parser = argparse.ArgumentParser(description='Generate additional documentation for ablate.dev')
    parser.add_argument('--componentListSource', dest='component_list_source', type=pathlib.Path,
                        help='The path to the markdown file with all components.')
    parser.add_argument('--componentOutputDir', dest='component_output_dir', type=pathlib.Path,
                        help='Optional path to output directory')
    args = parser.parse_args()

    # create component markdown
    if args.component_list_source:
        split_component_list(args.component_list_source, args.component_output_dir)
    else:
        print("no!!")


if __name__ == "__main__":
    parse()
