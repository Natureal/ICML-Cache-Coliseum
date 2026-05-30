from data_trace.data_trace import DataTrace
from utils.aligner import ShiftAligner, NormalAligner
from cache.cache import Cache, BoostCache, DumpCache
from cache.evict import *
from cache.hash import ShiftHashFunction, BrightKiteHashFunction, CitiHashFunction
from functools import partial
from typing import Tuple
from prettytable import PrettyTable
import tqdm
import copy
import argparse
import os
import pickle
import json
import csv
from pathos.multiprocessing import ProcessingPool as Pool

def process_cache(cache):
    with DataTrace(file_path) as trace:
        while not trace.done():
            pc, address = trace.next()
            cache.access(pc, address)
        stat = cache.stat()
        counts = None
        if hasattr(cache, 'layer_request_counts'):
            counts = cache.layer_request_counts()
        if counts is None:
            return stat
        x1, x2 = counts
        return (stat[0], stat[1], stat[2], stat[3], x1, x2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='xalanc')
    parser.add_argument("--test_all", action='store_true')

    parser.add_argument("--device", type=str, default='cpu')

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--oracle', action='store_true')
    mode_group.add_argument('--real', action='store_true')

    parser.add_argument('--pred', nargs='+', default='none', choices=['parrot', 'pleco', 'popu', 'pleco-bin', 'gbm', 'oracle_bin', 'oracle_dis'])

    parser.add_argument("--noise_type", type=str, default='logdis', choices=['dis', 'bin', 'logdis'])

    parser.add_argument("--dump_file", action='store_true')
    parser.add_argument("--output_root_dir", type=str, default='res')

    parser.add_argument("--verbose", action='store_true')

    parser.add_argument("--boost", action='store_true')
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--boost_fr", action='store_true')
    parser.add_argument("--boost_preds_dir", type=str, default='boost_traces')

    parser.add_argument("--model_fraction", type=str, default='1')
    parser.add_argument("--checkpoints_root_dir", type=str, default='checkpoints')
    parser.add_argument("--parrot_config_path", type=str, default='checkpoints/parrot/model_config.json')
    parser.add_argument("--lightgbm_config_path", type=str, default='checkpoints/lightgbm/model_config.json')

    args = parser.parse_args()
    file_path = f'traces/{args.dataset}/{args.dataset}_test.csv'
    if args.test_all:
        if args.dataset == 'brightkite' or args.dataset == 'citi':
            file_path = f'traces/{args.dataset}/{args.dataset}_all.csv'
    if args.dataset == 'brightkite':
        cache_line_size = 1
        capacity = 1000
        associativity = 10
        align_type = NormalAligner
        hash_type = BrightKiteHashFunction
    elif args.dataset == 'citi':
        cache_line_size = 1
        capacity = 1200
        associativity = 100
        align_type = NormalAligner
        hash_type = CitiHashFunction
    else:
        cache_line_size = 64
        capacity = 2097152
        associativity = 16
        align_type = ShiftAligner
        hash_type = ShiftHashFunction

    this_preds = []
    real_predictors_type = ['parrot', 'pleco', 'popu', 'pleco-bin', 'gbm']
    oracle_predictors_type = ['oracle_bin', 'oracle_dis']
    input_preds = args.pred
    if 'none' in input_preds:
        if args.real:
            this_preds = real_predictors_type
        else:
            this_preds = oracle_predictors_type
    else:
        this_preds = input_preds
    
    ###########################################################
    parrot_gen = gbm_gen = None
    ckpt_root_dir = args.checkpoints_root_dir
    if 'parrot' in this_preds:
        from model.models import ParrotModel
        from model import device_manager

        device_manager.set_device(args.device)
        this_dir = os.path.join(ckpt_root_dir, 'parrot', args.dataset, args.model_fraction)
        if not os.path.exists(this_dir):
            raise ValueError(f'Benchmark: {this_dir} not found checkpoints')
        this_ckpt_path = os.path.join(this_dir, f'best.ckpt')
        if not os.path.exists(this_ckpt_path):
            raise ValueError(f'Benchmark: {this_ckpt_path} not found checkpoints')
        print('Parrot Model Checkpoint:', this_ckpt_path)
        print('Parrot Model Fraction:', args.model_fraction)
        print('Parrot Model Config Path:', args.parrot_config_path)
        parrot_gen = lambda : ParrotModel.from_config(args.parrot_config_path, this_ckpt_path)
    if 'gbm' in this_preds:
        from model.models import LightGBMModel

        with open(args.lightgbm_config_path, "r") as f:
            model_config = json.load(f)
            deltanums = model_config['delta_nums']
            edcnums = model_config['edc_nums']

        this_dir = os.path.join(ckpt_root_dir, 'lightgbm', args.dataset, args.model_fraction)
        if not os.path.exists(this_dir):
            raise ValueError(f'Benchmark: {this_dir} not found checkpoints')
        this_ckpt_path = os.path.join(this_dir, f'{args.dataset}_{args.model_fraction}_{deltanums}_{edcnums}.txt')
        if not os.path.exists(this_ckpt_path):
            raise ValueError(f'Benchmark: {this_ckpt_path} not found checkpoints')

        threshold = 0.5
        threshold_path = os.path.join(this_dir, 'threshold')
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as file:
                content = file.read().strip()
                threshold = float(content)
        print(f'LightGBM: Fraction [{args.model_fraction}], Threshold [{threshold}], Model Checkpoint[{this_ckpt_path}], Delta[{deltanums}], EDC[{edcnums}]')
        gbm_gen = lambda : LightGBMModel.from_config(deltanums, edcnums, this_ckpt_path, threshold)
    
    print("Benchmark: Use Predictor:", this_preds)
    print('Benchmark: Use Trace:', file_path)
    if args.dump_file:
        print('Benchmark: Output Path:', args.output_root_dir)
    else:
        print('Benchmark: Only print')
    if args.boost:
        if args.real:
            print('Benchmark: Use Boost Trace Prediction')
        else:
            print('Benchmark: Use MultiProcess Boost')
    if args.boost_fr:
        print('Benchmark: Enable F&R Boost')
    online_types = [
        RandAlgorithm,
        LRUAlgorithm,
        MarkerAlgorithm,
        #partial(RDMAlgorithm, max_support_factor=5),
        partial(RDMAlgorithm, max_support_factor=20),
        #partial(RDMAlgorithm, max_support_factor=20),
        #partial(OnlineMinAlgorithm, max_support_factor=3),
        partial(OnlineMinAlgorithm, max_support_factor=1000),
        partial(OnlineMinRandomAlgorithm, max_support_factor=1000),
    ]

    sorted = False
    verbose = args.verbose

    func_dict = {}
    def register_func(this_partial, noise, baseline=False):
        if baseline:
            func_dict['OPT'] = {}
            func_dict['OPT'][0] = this_partial
        else:
            pretty_name = pretty_print(this_partial, verbose)
            if pretty_name not in func_dict:
                func_dict[pretty_name] = {}
            func_dict[pretty_name][noise] = this_partial
    
    register_func(PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'OracleDis'), 0, True)

    combiner_types = []

    boost_preds_dict = {}

    if not os.path.exists(args.boost_preds_dir):
        os.makedirs(args.boost_preds_dir)
    def boost_generate_prediction(pred_type, **kwargs):
        pred_algorithm = PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, pred_type, **kwargs)

        if args.test_all and (args.dataset == 'brightkite' or args.dataset == 'citi'):
            pred_pickle_path = os.path.join(args.boost_preds_dir, f'{args.dataset}_all_{pred_type}_{args.model_fraction}.pkl')
        else:
            pred_pickle_path = os.path.join(args.boost_preds_dir, f'{args.dataset}_{pred_type}_{args.model_fraction}.pkl')
        if not os.path.exists(pred_pickle_path):
            print(f'Boost Prediction: Generating Prediction for {pred_type}, Path: {pred_pickle_path}')
            if pred_type.endswith('State'):
                is_state = True
            else:
                is_state = False
            dump_cache = DumpCache(is_state, file_path, align_type, pred_algorithm, hash_type, cache_line_size, capacity, associativity)
            with DataTrace(file_path) as trace:
                with tqdm.tqdm(desc="Producing cache on Boost Prediction") as pbar:
                    while not trace.done():
                        pc, address = trace.next()
                        dump_cache.simulate(pc, address)
                        pbar.update(1) 
            lst = dump_cache.dump()
            with open(pred_pickle_path, 'wb') as f:
                pickle.dump(lst, f)
            return lst
        else:
            print(f'Boost Prediction: Find boost prediction for {pred_type}, Path: {pred_pickle_path}')
            with open(pred_pickle_path, 'rb') as f:
                lst = pickle.load(f)
                return lst

    if args.real:
        verbose = True

        ##########################################
        if 'parrot' in this_preds:
            if args.boost:
                boost_preds_dict['Parrot'] = boost_generate_prediction('Parrot', shared_model=parrot_gen())
                boost_preds_dict['Parrot-State'] = boost_generate_prediction('Parrot-State', associativity=associativity, shared_model=parrot_gen())

            online_types.extend([
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'Parrot', shared_model=parrot_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictiveMarker, 'Parrot', shared_model=parrot_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(LMarker, 'Parrot', shared_model=parrot_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(LNonMarker, 'Parrot', shared_model=parrot_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(FollowerRobust, boost=args.boost_fr), 'Parrot-State', associativity=associativity, shared_model=parrot_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'Parrot', shared_model=parrot_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'Parrot', shared_model=parrot_gen()),
            ])
            combiner_types.extend([
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'Parrot', shared_model=parrot_gen()), MarkerAlgorithm]),
                (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'Parrot', shared_model=parrot_gen()), MarkerAlgorithm]),
            ])


        ##########################################
        if 'pleco' in this_preds:
            if args.boost:
                boost_preds_dict['PLECO'] = boost_generate_prediction('PLECO')
                boost_preds_dict['PLECO-State'] = boost_generate_prediction('PLECO-State', associativity=associativity)

            online_types.extend([
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictiveMarker, 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(LMarker, 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(LNonMarker, 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveOnlineMinAlgorithm, max_support_factor=100), 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBOnlineMinAlgorithm, pred_budget=0, max_support_factor=100), 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBOnlineMinAlgorithm, pred_budget=1, max_support_factor=100), 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBOnlineMinAlgorithm, pred_budget=2, max_support_factor=100), 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBNewOnlineMinAlgorithm, pred_budget=0, max_support_factor=100), 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBNewOnlineMinAlgorithm, pred_budget=1, max_support_factor=100), 'PLECO'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBNewOnlineMinAlgorithm, pred_budget=2, max_support_factor=100), 'PLECO'),

                PredictAlgorithmFactory.generate_predictive_algorithm(partial(FollowerRobust, boost=args.boost_fr), 'PLECO-State', associativity=associativity),
                #PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'PLECO'),
                #PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'PLECO'),
            ])
            combiner_types.extend([
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO'), LRUAlgorithm]),
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO'), RDMAlgorithm]),
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO'), MarkerAlgorithm]),
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO'), OnlineMinAlgorithm]),
                #(partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO'), MarkerAlgorithm]),
            ])

        ##########################################
        if 'popu' in this_preds:
            if args.boost:
                boost_preds_dict['POPU'] = boost_generate_prediction('POPU')
                boost_preds_dict['POPU-State'] = boost_generate_prediction('POPU-State', associativity=associativity)

            online_types.extend([
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictiveMarker, 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(LMarker, 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(LNonMarker, 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveOnlineMinAlgorithm, max_support_factor=100), 'POPU'),
                #PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBOnlineMinAlgorithm, max_support_factor=100), 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBOnlineMinAlgorithm, pred_budget=0, max_support_factor=100), 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBOnlineMinAlgorithm, pred_budget=1, max_support_factor=100), 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBOnlineMinAlgorithm, pred_budget=2, max_support_factor=100), 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBNewOnlineMinAlgorithm, pred_budget=0, max_support_factor=100), 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBNewOnlineMinAlgorithm, pred_budget=1, max_support_factor=100), 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBNewOnlineMinAlgorithm, pred_budget=2, max_support_factor=100), 'POPU'),
                # PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBNewOnlineMinAlgorithm, pred_budget=4, max_support_factor=100), 'POPU'),
                # PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBNewOnlineMinAlgorithm, pred_budget=8, max_support_factor=100), 'POPU'),
                # PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveRPBNewOnlineMinAlgorithm, pred_budget=16, max_support_factor=100), 'POPU'),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(FollowerRobust, boost=args.boost_fr), 'POPU-State', associativity=associativity),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'POPU'),
                #PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'POPU'),
            ])
            combiner_types.extend([
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'POPU'), LRUAlgorithm]),
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'POPU'), RDMAlgorithm]),
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'POPU'), MarkerAlgorithm]),
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'POPU'), OnlineMinAlgorithm]),
                #(partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'POPU'), MarkerAlgorithm]),
            ])
        
        ##########################################
        if 'pleco-bin' in this_preds:
            if args.boost:
                boost_preds_dict['PLECO-Bin'] = boost_generate_prediction('PLECO-Bin', threshold=0.5)

            online_types.extend([
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO-Bin', threshold=0.5),
                PredictAlgorithmFactory.generate_predictive_algorithm(Mark0, 'PLECO-Bin', threshold=0.5),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(PredictiveOnlineMinAlgorithm, max_support_factor=100), 'PLECO-Bin', threshold=0.5),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'PLECO-Bin', threshold=0.5),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'PLECO-Bin', threshold=0.5),
            ])
            combiner_types.extend([
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO-Bin', threshold=0.5), MarkerAlgorithm]),
                (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'PLECO-Bin', threshold=0.5), MarkerAlgorithm]),
            ])

        ##########################################
        if 'gbm' in this_preds:
            if args.boost:
                boost_preds_dict['GBM'] = boost_generate_prediction('GBM', shared_model=gbm_gen())

            online_types.extend([
                PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'GBM', shared_model=gbm_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(Mark0, 'GBM', shared_model=gbm_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'GBM', shared_model=gbm_gen()),
                PredictAlgorithmFactory.generate_predictive_algorithm(partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'GBM', shared_model=gbm_gen()),
            ])
            combiner_types.extend([
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'GBM', shared_model=gbm_gen()), MarkerAlgorithm]),
                (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [PredictAlgorithmFactory.generate_predictive_algorithm(PredictAlgorithm, 'GBM', shared_model=gbm_gen()), MarkerAlgorithm]),
            ])

        ########################################################
        for online_type in online_types:
            register_func(online_type, 0)

        oracle_types = [
            # (PredictAlgorithm, "OracleDis")
        ]

        for oracle_type, pred_type_str in oracle_types:
            register_func(PredictAlgorithmFactory.generate_predictive_algorithm(oracle_type, pred_type_str), 0)

        for combiner, algs in combiner_types:
            if len(algs) > 1:
                this_partial = copy.deepcopy(combiner)
                this_partial.keywords['candidate_algorithms'] = algs
                register_func(this_partial, 0)
    else:
        noise_type = args.noise_type
        oracle_types = []
        combiner_types = []

        for online_type in online_types:
            register_func(online_type, 0)

        oracle_logdis_noise_mask = [0, 5, 10, 20, 30, 40, 50]
        oracle_dis_noise_mask = [0, 50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000]
        oracle_bin_noise_mask = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        if 'oracle_dis' in this_preds:
            oracle_types.extend([
                (PredictAlgorithm, 'OracleDis'),
                (PredictiveMarker, 'OracleDis'),
                (LMarker, 'OracleDis'),
                (LNonMarker, 'OracleDis'),
                (partial(PredictiveOnlineMinAlgorithm, max_support_factor=100), 'OracleDis'),
                (partial(PredictiveRPBOnlineMinAlgorithm, pred_budget=0, max_support_factor=100), 'OracleDis'),
                (partial(PredictiveRPBOnlineMinAlgorithm, pred_budget=1, max_support_factor=100), 'OracleDis'),
                (partial(PredictiveRPBOnlineMinAlgorithm, pred_budget=2, max_support_factor=100), 'OracleDis'),
                (partial(PredictiveRPBNewOnlineMinAlgorithm, pred_budget=0, max_support_factor=100), 'OracleDis'),
                (partial(PredictiveRPBNewOnlineMinAlgorithm, pred_budget=1, max_support_factor=100), 'OracleDis'),
                (partial(PredictiveRPBNewOnlineMinAlgorithm, pred_budget=2, max_support_factor=100), 'OracleDis'),

                (partial(FollowerRobust, boost=args.boost_fr), 'OracleState'),
                #(partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'OracleDis'),
                #(partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'OracleDis'),
            ])
            combiner_types.extend([
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleDis'), LRUAlgorithm]),
                # (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleDis'), RDMAlgorithm]),
                # (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleDis'), OnlineMinAlgorithm]),
                # (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleDis'), MarkerAlgorithm]),
                #(partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleDis'), RDMAlgorithm]),
                # (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleDis'), LRUAlgorithm]),
                # (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleDis'), RDMAlgorithm]),
                # (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleDis'), OnlineMinAlgorithm]),
                # (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleDis'), MarkerAlgorithm]),
            ])

        #####################################################
        if 'oracle_bin' in this_preds:
            oracle_types.extend([
                (PredictAlgorithm, 'OracleBin'),
                (Mark0, 'OracleBin'),
                (MarkAndPredict, 'OraclePhase'),
                (partial(PredictiveOnlineMinAlgorithm, max_support_factor=100), 'OracleBin'),
                (partial(Guard, follow_if_guarded=False, relax_times=0, relax_prob=0), 'OracleBin'),
                (partial(Guard, follow_if_guarded=False, relax_times=5, relax_prob=0), 'OracleBin'),
            ])
            combiner_types.extend([
                (partial(CombineDeterministicAlgorithm, switch_bound=1, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleBin'), MarkerAlgorithm]),
                (partial(CombineRandomAlgorithm, alpha=0.0, beta=0.99, lazy_evictor_type=LRUEvictor), [(PredictAlgorithm, 'OracleBin'), MarkerAlgorithm]),
            ])

        #####################################################
        for oracle_type, pred_type_str in oracle_types:
            if noise_type == 'dis':
                for noise in oracle_dis_noise_mask:
                    register_func(PredictAlgorithmFactory.generate_predictive_algorithm(oracle_type, pred_type_str, reuse_dis_noise_sigma=noise, lognormal=False, associativity=associativity), noise)
            elif noise_type == 'logdis':
                for noise in oracle_logdis_noise_mask:
                    register_func(PredictAlgorithmFactory.generate_predictive_algorithm(oracle_type, pred_type_str, reuse_dis_noise_sigma=noise, lognormal=True, associativity=associativity), noise)
            elif noise_type == 'bin':
                if pred_type_str == 'OracleBin' or pred_type_str == 'OraclePhase':
                    for noise in oracle_bin_noise_mask:
                        register_func(PredictAlgorithmFactory.generate_predictive_algorithm(oracle_type, pred_type_str, bin_noise_prob=noise, associativity=associativity), noise)
            else:
                raise ValueError('Invalid noise type')

        def mask_combiner(noise):
            for combiner, algs in combiner_types:
                candidate_algorithms = []
                for alg in algs:
                    if isinstance(alg, Tuple):
                        alg_type, pred_type_str = alg
                        if noise_type == 'dis':
                            candidate_algorithms.append(PredictAlgorithmFactory.generate_predictive_algorithm(alg_type, pred_type_str, associativity=associativity, reuse_dis_noise_sigma=noise, lognormal=False))
                        elif noise_type == 'logdis':
                            candidate_algorithms.append(PredictAlgorithmFactory.generate_predictive_algorithm(alg_type, pred_type_str, associativity=associativity, reuse_dis_noise_sigma=noise, lognormal=True))
                        elif noise_type == 'bin':
                            if pred_type_str == 'OracleBin' or pred_type_str == 'OraclePhase':
                                candidate_algorithms.append(PredictAlgorithmFactory.generate_predictive_algorithm(alg_type, pred_type_str, associativity=associativity, bin_noise_prob=noise))
                        else:
                            raise ValueError('Invalid noise type')
                    else:
                        candidate_algorithms.append(alg)
                
                if len(candidate_algorithms) > 1:
                    this_partial = copy.deepcopy(combiner)
                    this_partial.keywords['candidate_algorithms'] = candidate_algorithms
                    register_func(this_partial, noise)

        if noise_type == 'bin':
            for noise in oracle_bin_noise_mask:
                mask_combiner(noise)
        elif noise_type == 'logdis':
            for noise in oracle_logdis_noise_mask:
                mask_combiner(noise)
        elif noise_type == 'dis':
            for noise in oracle_dis_noise_mask:
                mask_combiner(noise)
        else:
            raise ValueError('Invalid noise type')

###############################################################
    cache_dict = {}
    caches = []
    funcs = [(outer_key, inner_key, value) for outer_key, inner_dict in func_dict.items() for inner_key, value in inner_dict.items()]

    boost_cls_list = {}
    for name, lst, in boost_preds_dict.items():
        if name in PredictAlgorithmFactory.predictor_evict_dict:
            this_cls = PredictAlgorithmFactory.predictor_evict_dict[name][1]
            boost_cls_list[this_cls] = lst

    with tqdm.tqdm(desc="Init caches for benchmark", total=len(funcs)) as pbar:
        def register_cache(pretty_name, noise, cache):
            if pretty_name not in cache_dict:
                cache_dict[pretty_name] = {}
            cache_dict[pretty_name][noise] = cache

        def judge_pred_type(this_partial):
            if hasattr(this_partial, 'keywords'):
                kw = this_partial.keywords
                if 'predictor_type' in kw:
                    predictor_type = kw['predictor_type']
                    pred_cls_type = predictor_type.func if hasattr(predictor_type, 'func') else predictor_type
                    return pred_cls_type
            return None

        for pretty_name, noise, this_partial in funcs:
            if args.oracle:
                if args.boost:
                    # todo
                    cache = Cache(file_path, align_type, this_partial, hash_type, cache_line_size, capacity, associativity)
                else:
                    cache = Cache(file_path, align_type, this_partial, hash_type, cache_line_size, capacity, associativity)
            elif args.real:
                cache = None
                if args.boost:
                    pred_cls_type = judge_pred_type(this_partial)
                    if pred_cls_type is not None:
                        if pred_cls_type in boost_cls_list:
                            tqdm.tqdm.write(f"Enbale Boost for {pretty_name} --> {pred_cls_type.__name__}")
                            cache = BoostCache(copy.deepcopy(boost_cls_list[pred_cls_type]), file_path, align_type, this_partial, hash_type, cache_line_size, capacity, associativity)

                    if hasattr(this_partial, 'keywords'):
                        kw = this_partial.keywords
                        if 'candidate_algorithms' in kw:
                            algs = kw['candidate_algorithms']
                            for alg in algs:
                                pred_cls_type = judge_pred_type(alg)
                                if pred_cls_type in boost_cls_list:
                                    tqdm.tqdm.write(f"Enbale Boost for {pretty_name} --> {pred_cls_type.__name__}")
                                    cache = BoostCache(copy.deepcopy(boost_cls_list[pred_cls_type]), file_path, align_type, this_partial, hash_type, cache_line_size, capacity, associativity)
                                    break
                if cache is None:
                    cache = Cache(file_path, align_type, this_partial, hash_type, cache_line_size, capacity, associativity)
            register_cache(pretty_name, noise, cache)
            caches.append(cache)
            pbar.update(1)

    if args.boost and args.oracle:
        num_workers = args.num_workers
        if num_workers == -1:
            num_workers = os.cpu_count()
        
        print(f'Benchmark: Workers[{num_workers}]')
        with Pool(processes=num_workers) as pool:
            stats = list(tqdm.tqdm(pool.map(process_cache, caches), total=len(caches)))
        for i, stat in enumerate(stats):
            caches[i].set_stat(stats[i][0], stats[i][1], stats[i][2])
            if len(stat) >= 6 and hasattr(caches[i], 'set_layer_request_counts'):
                caches[i].set_layer_request_counts(stat[4], stat[5])
    else:
        with DataTrace(file_path) as trace:
            with tqdm.tqdm(desc="Producing cache on MemoryTrace") as pbar:
                while not trace.done():
                    pc, address = trace.next()
                    for cache in caches:
                        cache.access(pc, address)
                    pbar.update(1)

        # Aggregate and store OnlineMin layer request counts (single-process runs).
        for cache in caches:
            if hasattr(cache, 'layer_request_counts') and hasattr(cache, 'set_layer_request_counts'):
                counts = cache.layer_request_counts()
                if counts is not None:
                    cache.set_layer_request_counts(counts[0], counts[1])

    table = PrettyTable() 

    opt_miss = np.inf
    if 'OPT' in cache_dict:
        hit, opt_miss, total, rate = cache_dict['OPT'][0].stat()

    if verbose:
        table.field_names = ["Name", "Hit", "Miss", "Total", "Hit Rate", "Competitive Ratio"]
        for i, (pretty_name, _, _) in enumerate(funcs):
            hit, miss, total, rate = caches[i].stat()
            table.add_row([pretty_name, hit, miss, total, rate, f"{miss / opt_miss:.3f}"])
        if sorted:
            table.sortby = "Hit Rate"
            table.reversesort = True
    else:
        if noise_type == 'dis':
            table.field_names = ["Name"] + [f'Dis-{noise}' for noise in oracle_dis_noise_mask]
        elif noise_type == 'logdis':
            table.field_names = ["Name"] + [f'LogDis-{noise}' for noise in oracle_logdis_noise_mask]
        elif noise_type == 'bin':
            table.field_names = ["Name"] + [f'Bin-{noise}' for noise in oracle_bin_noise_mask]
        for pretty_name in cache_dict:
            if pretty_name == 'OPT':
                continue
            lst = [pretty_name]
            for noise in cache_dict[pretty_name]:
                _, miss, _, rate = cache_dict[pretty_name][noise].stat()
                lst.append(f'{rate:.3f}/{miss / opt_miss:.3f}')
            if len(lst) == 2:
                lst.extend([lst[-1]] * (len(table.field_names) - 2))
            table.add_row(lst)

    def write_table_csv(path):
        with open(path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(table.field_names)
            writer.writerows(table._rows)

    if args.dump_file:
        res_dir = os.path.join(args.output_root_dir, args.dataset, args.model_fraction)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        if args.real:
            write_table_csv(os.path.join(res_dir, f"{'_'.join(this_preds)}.csv"))
        else:
            write_table_csv(os.path.join(res_dir, f"{args.noise_type}.csv"))
    # Print OnlineMin phase stats: x1=#(L0 requests), x2=#(non-L0 requests).
    for i, (pretty_name, noise, this_partial) in enumerate(funcs):
        this_cls = this_partial
        if hasattr(this_partial, 'func'):
            this_cls = this_partial.func
        if this_cls is OnlineMinAlgorithm and noise == 0:
            x1 = getattr(caches[i], 'layer0_requests', None)
            x2 = getattr(caches[i], 'non_layer0_requests', None)
            if x1 is None or x2 is None:
                continue
            ratio = float('inf') if x1 == 0 else (x1 + x2) / x1
            print(f'OnlineMin layer stats: x1={x1}, x2={x2}, (x1+x2)/x1={ratio}')

    print(table)
            
