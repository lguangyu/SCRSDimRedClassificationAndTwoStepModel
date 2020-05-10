#!/usr/bin/env python3

import argparse
import io
import numpy
import sys


def get_args():
	dfl_lb = 400.0
	dfl_ub = 1800.0
	dfl_bs = 2.0

	ap = argparse.ArgumentParser()
	ap.add_argument("input", type = str, nargs = "?", default = "-",
		help = "input 2-column spectrum .txt dump; read from stdin if empty or "
			"'-'")
	ap.add_argument("--min-wavenumber", type = float, default = dfl_lb,
		metavar = "float",
		help = "lower boundary of binning wavenumber window (default: %.1f)"\
			% dfl_lb)
	ap.add_argument("--max-wavenumber", type = float, default = dfl_ub,
		metavar = "float",
		help = "upper boundary of binning wavenumber window (default: %.1f)"\
			% dfl_ub)
	ap.add_argument("-b", "--bin-size", type = float, default = dfl_bs,
		metavar = "float",
		help = "size of bin window (default: %.1f)" % dfl_bs)
	ap.add_argument("-o", "--output", type = str, default = "-",
		metavar = "file",
		help = "output file; write to stdout if empty or '-'")
	# parse and refine args
	args = ap.parse_args()
	if args.input == "-":
		args.input = sys.stdin
	if args.output == "-":
		args.output = sys.stdout
	return args


def get_fp(fn_or_fp, *ka, factory = open, **kw):
	if isinstance(fn_or_fp, io.IOBase):
		return fn_or_fp
	elif isinstance(fn_or_fp, str):
		return factory(fn_or_fp, *ka, **kw)
	else:
		raise TypeError("fn_or_fp must be io.IOBase or str, not '%s'"\
			% type(fn_or_fp).__name__)


def load_data(file, delimiter = "\t"):
	data = numpy.loadtxt(file, delimiter = delimiter, dtype = float)
	wavenumbers = data[:, 0]
	intensities = data[:, 1]
	return wavenumbers, intensities


def bin(wavenumbers, intensities, *, wn_min, wn_max, bin_size):
	ret_wn = numpy.arange(wn_min, wn_max, bin_size, dtype = numpy.float64)
	ret_ts = numpy.zeros(len(ret_wn), dtype = numpy.float64)
	for i, wn in enumerate(ret_wn):
		mask = numpy.logical_and(wavenumbers >= wn, wavenumbers < wn + bin_size)
		ret_ts[i] = 0 if not mask.any() else intensities[mask].mean()
	# set new wavenumber to the bin window centeroid
	ret_wn += (bin_size / 2)
	return ret_wn, ret_ts


def main():
	args = get_args()
	# load data
	with get_fp(args.input, "r") as fp:
		wavenumbers, intensities = load_data(fp)
	# bin
	bin_wavenumbers, bin_intensities = bin(wavenumbers, intensities,
		wn_min = args.min_wavenumber,
		wn_max = args.max_wavenumber,
		bin_size = args.bin_size)
	# output
	arr = numpy.hstack([
		bin_wavenumbers.reshape(-1, 1),
		bin_intensities.reshape(-1, 1)
	])
	with get_fp(args.output, "w") as fp:
		numpy.savetxt(fp, arr, delimiter = "\t", fmt = "%f")
	return


if __name__ == "__main__":
	main()
