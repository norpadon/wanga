import jinja2
import jinja2.meta

__all__ = [
    "make_template",
    "get_local_variables",
]


BUILTIN_FUNCTIONS = {zip, enumerate, range}


ENVIROMENT = jinja2.Environment(autoescape=False, undefined=jinja2.StrictUndefined)
ENVIROMENT.globals.update({function.__name__: function for function in BUILTIN_FUNCTIONS})


def make_template(template_string: str) -> jinja2.Template:
    r"""Create a Jinja2 template from a string."""
    return ENVIROMENT.from_string(template_string)


def get_local_variables(template_string: str) -> list[str]:
    r"""Get the local variables mentioned in a Jinja2 template."""
    ast = ENVIROMENT.parse(template_string)
    return list(jinja2.meta.find_undeclared_variables(ast) - ENVIROMENT.globals.keys())
