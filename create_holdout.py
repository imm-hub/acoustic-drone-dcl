import shutil
from pathlib import Path
import random

random.seed(42)

# Source folders
drone_src = Path('data/external/acolab/drone')
bg_src = Path('data/external/acolab/no_drone')

# Create test folders
Path('data/external/acolab_test/drone').mkdir(parents=True, exist_ok=True)
Path('data/external/acolab_test/no_drone').mkdir(parents=True, exist_ok=True)

# Move 20% of drone files to test
drone_files = list(drone_src.glob('*.wav'))
random.shuffle(drone_files)
test_count = int(len(drone_files) * 0.2)

print(f"Moving {test_count} drone files to test set...")
for f in drone_files[:test_count]:
    shutil.move(str(f), f'data/external/acolab_test/drone/{f.name}')

# Move 20% of background files to test
bg_files = list(bg_src.glob('*.wav'))
random.shuffle(bg_files)
test_count_bg = int(len(bg_files) * 0.2)

print(f"Moving {test_count_bg} background files to test set...")
for f in bg_files[:test_count_bg]:
    shutil.move(str(f), f'data/external/acolab_test/no_drone/{f.name}')

# Show counts
print(f"\nTraining set (acolab):")
print(f"  Drone: {len(list(drone_src.glob('*.wav')))}")
print(f"  Background: {len(list(bg_src.glob('*.wav')))}")
print(f"\nTest set (acolab_test) - HOLD OUT:")
print(f"  Drone: {len(list(Path('data/external/acolab_test/drone').glob('*.wav')))}")
print(f"  Background: {len(list(Path('data/external/acolab_test/no_drone').glob('*.wav')))}")