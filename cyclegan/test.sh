#!./scripts/test_cyclegan.sh

python3 test.py --dataroot path/to/vein_dataset \
                --name vein_cyclegan \
                --model cycle_gan \
                --dataset_mode single 
