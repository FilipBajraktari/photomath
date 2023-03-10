# 82 simbola
#TODO: treba smanjiti slova, div->frac
niz = [
    '!', '(', ')', '+', ',', '-', '=', '[', ']', '{', '}', '0',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'alpha', 'ascii_124',
    'b', 'beta', 'C', 'cos', 'd', 'Delta', 'div', 'e', 'exists', 'f', 'forall', 'forward_slash',
    'G', 'gamma', 'geq', 'gt', 'H', 'i', 'in', 'infty', 'int', 'j', 'k', 'l',
    'lambda', 'ldots', 'leq', 'lim', 'log', 'lt', 'M', 'mu', 'N', 'neq', 'o', 'p',
    'phi', 'pi', 'pm', 'prime', 'q', 'R', 'rightarrow', 'S', 'sigma', 'sin', 'sqrt', 'sum',
    'T', 'tan', 'theta', 'times', 'u', 'v', 'w', 'X', 'y', 'z'
]


def number2symbol(nbr):
    return niz[nbr]


def symbol2number(symbol):
    return niz.index(symbol)


def exist(symbol):
    return (symbol in niz)