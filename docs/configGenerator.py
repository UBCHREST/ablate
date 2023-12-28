import argparse
import pathlib

# define some required strings
TemplateStart = "{{"
TemplateEnd = "}}"


# create a copy and update template for all files that have {{TEMPLATE}} code in title
def generate_config_files(root_directory, new_directory, replacements):
    for path in root_directory.rglob(f'*{TemplateStart}*{TemplateEnd}*'):
        # get the relative path
        relative_path = path.relative_to(root_directory)

        # create the new name
        new_path_name = str(relative_path)

        # read in the entire file
        content = path.read_text()

        # replace the files in the name and content
        for key, value in replacements.items():
            new_path_name = new_path_name.replace(f'{TemplateStart}{key}{TemplateEnd}', value)
            content = content.replace(f'{TemplateStart}{key}{TemplateEnd}', value)

        # make sure it is a new path
        if new_path_name == str(relative_path):
            continue

        # create a new file and write the contents to it
        new_path = new_directory / new_path_name

        # create the parent directory if needed
        new_path.parent.mkdir(parents=True, exist_ok=True)

        # write the file
        new_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate config files specific to this build of PETSc and ABLATE')
    parser.add_argument('--configDirectory', dest='config_directory', type=pathlib.Path,
                        help='The path to the root of the config directory')
    parser.add_argument('--outputDirectory', dest='output_directory', type=pathlib.Path,
                        help='The path to the root of the output directory')
    args = parser.parse_args()

    # create component markdown
    if args.config_directory:
        generate_config_files(args.config_directory, args.output_directory, {'PETSC_COMMIT': 'abc'})
