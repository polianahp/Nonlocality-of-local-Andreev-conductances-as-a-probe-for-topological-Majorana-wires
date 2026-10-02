import pyflakes.api
import pyflakes.reporter
import sys

with open('cut_analysis.py', 'r') as f:
    code = f.read()

reporter = pyflakes.reporter.Reporter(sys.stdout, sys.stderr)
pyflakes.api.check(code, 'cut_analysis.py', reporter)
