from naslib.search_spaces.darts import MacroGraph, PRIMITIVES, OPS
from naslib.optimizers.oneshot.darts import DARTSOptimizer, Searcher
from naslib.utils import config_parser


if __name__ == '__main__':
    config = config_parser('../../configs/default.yaml')

    one_shot_optimizer = DARTSOptimizer.from_config(**config)
    search_space = MacroGraph.from_optimizer_op(
        one_shot_optimizer,
        config=config,
        primitives=PRIMITIVES,
        ops_dict=OPS
    )
    one_shot_optimizer.init()

    searcher = Searcher(search_space, arch_optimizer=one_shot_optimizer)
    searcher.run()
