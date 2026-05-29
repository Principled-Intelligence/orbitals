import typer

from orbitals.scope_guard_v2 import ScopeGuardV2

app = typer.Typer()


@app.command("convert-default-model-name")
def convert_default_model_name(model_name: str = typer.Argument(..., help="The model name")):
    print(ScopeGuardV2.maybe_map_model(model_name))
