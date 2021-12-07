from great_expectations.cli import suite
import os
ANNOTATION_FOLDER = 'dataset/VisDrone2020-CC/raw/annotations'

if __name__ == '__main__':
    files = os.listdir(ANNOTATION_FOLDER)
    files = map(lambda x: os.path.join(ANNOTATION_FOLDER, x), files)
    for file in files:
        suite.suite_new()
        cli.main(config_file='great_expectations/great_expectations.yml')