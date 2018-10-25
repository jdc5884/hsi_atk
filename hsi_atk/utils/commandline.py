import argparse


class CommandLine:

    def __init__(self):
        self.arguments = self.check_args()
        self.print_args()

    @staticmethod
    def usage():
        options = ['launcher']
        print('Command-line options for hsi_atk')
        print('--help or -h to see the help menu')
        print('--verbose or -v for verbose mode')
        print('--output or -o to specify output directory')
        print('--folder or -f to specify parent directory of hsi files')
        print('--param or -p to specify parameters files (JSON or TXT)')
        print('\tParameters file specifies genotypes, packet #s, treatment options,')
        print('\toutput options, expected file types, and intermediary steps.')
        print('\tParameters file also specifies the analysis to be run.')

    def print_args(self):
        args_dict = vars(self.arguments)
        for k in args_dict:
            print(k.capitalize() + ':', args_dict[k])

    def check_args(self):
        parser = argparse.ArgumentParser(description='Commandline argparser for hsi_atk')
        parser.add_argument('-v', '--verbose', help='Trugger verbose mode', action='store', default=False)
        parser.add_argument('-o', '--output', help='Prefix for output files', type=str, default='merged_output')

        subparsers = parser.add_subparsers(help='Merge mode', dest='mode')
        self.add_options(subparsers)
        args = parser.parse_args()
        return args

    @property
    def args(self):
        return self.arguments

    def add_options(self, sub):
        self.add_launcher_options(sub)

    @staticmethod
    def add_launcher_options(sub):
        launcher_parser = sub.add_parser('launcher')
        launcher_parser.add_argument('-f', '--folder', help='The folder containing hyperspectral images', required=True,
                                     type=str)
        launcher_parser.add_argument('-p', '--param', help='The file containing options for hyperspectral analysis',
                                     required=True, type=str)