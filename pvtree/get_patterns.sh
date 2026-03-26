
# generate palmprint lines
python3 get_bezier.py --num_ids 8 --output ./bezierpalm

# generate palm vein patterns
python3 main.py --case 1 --num_percase 2 --num_sams 3  &
python3 main.py --case 2 --num_percase 2 --num_sams 3  &
python3 main.py --case 3 --num_percase 2 --num_sams 3  &
python3 main.py --case 4 --num_percase 2 --num_sams 3  

wait

# Preprocess: Blend with palmprint creases + intra-class augmentation.
python3 process.py
