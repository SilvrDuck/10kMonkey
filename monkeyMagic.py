from IPython.core.magic import register_cell_magic


out = []

@register_cell_magic
def cmagic(line, cell):
    out.append(cell)
    return line, cell

@register_cell_magic
def gmagic(line, cell):

    return out

