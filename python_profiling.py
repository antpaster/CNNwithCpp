import cProfile
import pstats
from vgg_example import training_loop  # your training loop entry point

with cProfile.Profile() as pr:
    training_loop()

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(20)
