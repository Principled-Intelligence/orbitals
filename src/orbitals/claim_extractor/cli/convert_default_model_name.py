import typer

from orbitals.claim_extractor import ClaimExtractor

app = typer.Typer()


@app.command("convert-default-model-name")
def convert_default_model_name(model_name: str = typer.Argument(..., help="The model name")):
    print(ClaimExtractor.maybe_map_model(model_name))
