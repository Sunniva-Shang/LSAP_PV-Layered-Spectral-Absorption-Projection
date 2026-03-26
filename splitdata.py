import shutil, os
import argparse
from tqdm import tqdm
def split(p, n):
    plist = os.listdir(p)
    l = len(plist)  
    ll = l // n    
    remainder = l % n  
    
    for i in range(n):
        dp = p + '_{}'.format(i)
        if not os.path.exists(dp):
            os.mkdir(dp)
        
        current_count = ll + (1 if i < remainder else 0)
        
        start_idx = i * ll + min(i, remainder)
        end_idx = start_idx + current_count
        
        for j in range(start_idx, end_idx):
            sp = os.path.join(p, plist[j])
            ddp = os.path.join(dp, plist[j])
            shutil.copytree(sp, ddp)
        
        print(f'{i}: {len(os.listdir(dp))}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='split')
    parser.add_argument('--p', type=str, default='./pvtree/pv_pattern_results/palmvein')
    parser.add_argument('--n', type=int, default=8)
    args = parser.parse_args()
    print('spliting data...')
    split(args.p, args.n)
    print(f'split the data into {args.n} parts!')
