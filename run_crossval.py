import argparse
import functools

import lib.utils as utils
import lib.traffic as trf

parser = argparse.ArgumentParser()
parser.add_argument("spec_input_file")
parser.add_argument("output_file")
parser.add_argument("--threads", type=int, required=False)
parser.add_argument("--progress", action='store_true')

args = parser.parse_args()

exec_func = utils.gzip_pickle_load(args.spec_input_file)
exec_args = {
    'traffic_or_log_provider': functools.partial(
        trf.build_traffic,
        traffic_full=utils.gzip_pickle_load('./traffic_full.gz'),
        charging_stations=utils.gzip_pickle_load('./charging_stations.gz'),
        station_distances_mtx=utils.gzip_pickle_load('./station_distances_mtx.gz'))
}

if args.threads is not None:
    exec_args['threads'] = args.threads
if args.progress:
    exec_args['progress'] = True

result = exec_func(**exec_args)
utils.gzip_pickle_dump(result, args.output_file)
