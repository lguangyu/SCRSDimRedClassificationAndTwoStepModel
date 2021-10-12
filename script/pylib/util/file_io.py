#!/usr/bin/env python3

import io
import json
import sys


def get_fh(file, mode = "r", *ka, open_factory = open, **kw) -> io.IOBase:
	"""
	return a file handle, wrapper function to ensure safe invocation when used
	used on an instance of file handle;

	ARGUMENTS
	file:
		of type io.IOBase (e.g. stderr) or a str (as file name); return <file>
		unchanged if <file> is an io.IOBase instance; no mode check will be
		performed;
	mode:
		open mode; omitted if <file> is an io.IOBase instance; this argument
		follows the convention used by open_factory() as the second positional
		argument if used;
	open_factory:
		method to create the file handle; default is open() from built-in;
	*ka, **kw:
		other keyargs/kwargs passed to open_factory()
	"""
	if isinstance(file, io.IOBase):
		fh = file
	elif isinstance(file, str):
		fh = open_factory(file, mode, *ka, **kw)
	else:
		raise TypeError("file must be io.IOBase or str, not '%s'"\
			% type(file).__name__)
	return fh


def save_as_json(obj, file, *, human_readable = False, **kw) -> None:
	"""
	serialize obj into file in json format;

	ARGUMENTS
	obj:
		an object can be serialized as json;
	file:
		of type io.IOBase (e.g. stderr) or a str (as file name);
	human_readable:
		if True, also pass indent='\\t' and 'sort_keys=True' to json.dump();
		this overrides those arguments repetitively passed in **kw;
	**kw:
		other kwargs passed to json.dump()
	"""
	if human_readable:
		kw.update({"indent": "\t", "sort_keys": True})
	with get_fh(file, "w") as fp:
		json.dump(obj, fp, **kw)
	return
