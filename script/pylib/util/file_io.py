#!/usr/bin/env python3

import sys
import io


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
