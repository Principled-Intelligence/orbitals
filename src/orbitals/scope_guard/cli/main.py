import typer

from . import convert_default_model_name, serve

app = typer.Typer()

app.add_typer(serve.app)
app.add_typer(convert_default_model_name.app)


def main():
    app()


if __name__ == "__main__":
    main()
