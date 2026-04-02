import typer

from orbitals.scope_guard import ScopeGuard

app = typer.Typer()


@app.command("convert-default-model-name")
def convert_default_model_name(model_name: str = typer.Argument(..., help="The model name")):
    print(ScopeGuard.maybe_map_model(model_name))
