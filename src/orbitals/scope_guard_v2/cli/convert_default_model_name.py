import typer

app = typer.Typer()


@app.command("convert-default-model-name")
def convert_default_model_name(model_name: str = typer.Argument(..., help="The model name")):
    print(model_name)
