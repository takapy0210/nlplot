def is_notebook() -> bool:
    """ Check if the execution environment is an interactive shell or a jupyter_notebook.
    Returns:
        True or False
    reference:
        https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        # get_ipython function is an interactive shell or,
        # in the case of jupyter_notebooks, an already-loaded
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
