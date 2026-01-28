import typer

from ..scope_guard.cli.main import app as scope_app

app = typer.Typer()

app.add_typer(scope_app, name="scope-guard")


def main():
    app()


if __name__ == "__main__":
    main()
