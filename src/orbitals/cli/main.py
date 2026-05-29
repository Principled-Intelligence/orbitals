import typer

from ..claim_extractor.cli.main import app as claim_extractor_app
from ..scope_guard.cli.main import app as scope_app
from ..scope_guard_v2.cli.main import app as scope_v2_app

app = typer.Typer()

app.add_typer(scope_app, name="scope-guard")
app.add_typer(scope_v2_app, name="scope-guard-v2")
app.add_typer(claim_extractor_app, name="claim-extractor")


def main():
    app()


if __name__ == "__main__":
    main()
