import click
import expr_tracker as et


@click.group()
def main():
    pass


@main.command()
@click.argument("msg")
@click.option("--title", default="Alert", help="Title of the alert")
@click.option("--level", default="info", help="Level of the alert")
def alert(msg: str, title: str = "Alert", level: str = "info"):
    et.alert(title=title, text=msg, level=level)
