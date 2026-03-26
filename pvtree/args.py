import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')

    parser.add_argument('--step', type=float, default=1.0)
    parser.add_argument('--step2d', type=float, default=1.0)
    parser.add_argument('--step2d_no', type=float, default=10000)
    parser.add_argument('--num_point', type=int, default=100)
    parser.add_argument('--x_range', type=int, default=70)
    parser.add_argument('--y_range', type=int, default=20)  
    parser.add_argument('--z_range', type=int, default=70)
    parser.add_argument('--r_ori', type=int, default=1.5)
    parser.add_argument('--ratioE', type=int, default=0.08, help='The radius is weighted by the number of terminal nodes at each point.') 
    parser.add_argument('--gamma', type=int, default=3.0)
    parser.add_argument('--ratioQ', type=int, default=0.5)

    parser.add_argument('--num_percase', type=int, default=10, help='Number of ID for each case.')
    parser.add_argument('--num_sams', type=int, default=3, help='Number of samples for each ID.')
    parser.add_argument('--case', type=int, default=2)

    parser.add_argument('--crop_path', type=str, default='pv_pattern_results/crop')
    parser.add_argument('--pv_path', type=str, default='pv_pattern_results/pv')
    parser.add_argument('--aug_path', type=str, default='pv_pattern_results/palmvein')
    parser.add_argument('--palmprint_path', type=str, default='./bezierpalm')
    parser.add_argument('--erapalmprint_path', type=str, default='./bezierpalm_era')

    args = parser.parse_args()
    return args
