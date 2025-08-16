import argparse
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))
from ingest_and_index import ingest_paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args()
    vs = ingest_paths(args.paths)
    print('Indexed')
