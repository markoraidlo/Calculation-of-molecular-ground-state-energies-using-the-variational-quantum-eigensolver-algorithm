import sys

def main():
    """
    -min molecule \n
    -scan counts molecule \n
    -min -scan counts molecule \n
    Molecules: H2, LiH, BeH2, H2O
    """
    #Program args
    args = sys.argv[1:]
    print(args)
    assert len(args) >= 2
    min_mode = ('-min' in args)
    scan_mode = ('-scan' in args)
    molecule_name = args[-1]
    if scan_mode:
        counts = int(args[-2])

    geometry_dict_min = {'H2': [[ 'H', [ 0, 0, 0]],
                                [ 'H', [ 0, 0, 0.74]]], 
                        'LiH': [['Li', [0, 0, 0]] ,
                                ['H', [0, 0, 1.5949]]],
                        'BeH2': [['Be', [ 0, 0, 0 ]],
                                ['H', [ 0, 0, 1.3264]],
                                ['H', [ 0, 0, -1.3264]]],
                        'H2O': [['O', [-0.053670056908, -0.039737675589, 0]],
                                ['H', [-0.028413670411,  0.928922556351, 0]],
                                ['H', [0.880196420813,  -0.298256807934, 0]]]}
    geometry_dict_scan = {}

    print(molecule_name)
    geometry = geometry_dict_min[molecule_name]
    print(geometry)
        

if __name__ == "__main__":
    main()