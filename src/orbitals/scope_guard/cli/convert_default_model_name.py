import typer

from orbitals.scope_guard import ScopeGuard

app = typer.Typer()


@app.command("convert-default-model-name")
def serve(model_name: str = typer.Argument(..., help="The model name")):
    print(ScopeGuard.maybe_map_model(model_name))
